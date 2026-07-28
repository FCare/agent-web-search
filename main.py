import asyncio
import json
import logging
import os
import sys

from urllib.parse import quote

import aiohttp
import openai
import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from nexus_client import NexusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# Authentik OAuth
AUTHENTIK_URL = os.environ.get("AUTHENTIK_URL", "https://sso.caronboulme.fr")
AUTHENTIK_CLIENT_ID = os.environ["AUTHENTIK_CLIENT_ID"]
AUTHENTIK_CLIENT_SECRET = os.environ["AUTHENTIK_CLIENT_SECRET"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]
DEFAULT_N_RESULTS = int(os.environ.get("DEFAULT_N_RESULTS", "4"))
FETCH_TOP_N = int(os.environ.get("FETCH_TOP_N", "4"))
# 3s plutôt que 8 : le fetch est parallèle, son coût est celui de la page la
# plus lente. Mesuré sur un lot de 8 pages, sept répondent en moins de 2s et
# une traîne — la borne basse coupe cette traînarde, dont on garde l'extrait du
# moteur, au lieu de faire attendre tout le monde.
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "3"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "3000"))
ANYSEARCH_API_KEY = os.environ.get("ANYSEARCH_API_KEY", "")
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"

# Backends ddgs interrogés dans l'ordre, le premier qui répond gagne. Mesuré le
# 28/07/2026 depuis cette IP : en text bing/yandex/yahoo répondent, brave,
# duckduckgo, google, mojeek et startpage renvoient systématiquement "No results
# found" ; en news bing et duckduckgo répondent, yahoo non. Les inutiles sont
# écartés pour ne pas payer un aller-retour à vide avant le backend qui marche.
DDGS_TEXT_BACKENDS = os.environ.get("DDGS_TEXT_BACKENDS", "bing,yandex,yahoo").split(",")
DDGS_NEWS_BACKENDS = os.environ.get("DDGS_NEWS_BACKENDS", "bing,duckduckgo").split(",")
DDGS_TIMEOUT = int(os.environ.get("DDGS_TIMEOUT", "20"))
DDGS_REGION = os.environ.get("DDGS_REGION", "fr-fr")

# API MediaWiki plutôt que le backend wikipedia de ddgs, qui fait du rapprochement
# de titre et non de la recherche : il répond "Guerre en Irak" à "guerre en Iran"
# et ne trouve rien pour une question formulée à l'oral. L'API officielle est
# gratuite, sans clé, et répond en 0,6 s.
WIKIPEDIA_API = os.environ.get("WIKIPEDIA_API", "https://fr.wikipedia.org/w/api.php")
WIKIPEDIA_N = int(os.environ.get("WIKIPEDIA_N", "3"))
WIKIPEDIA_UA = "caronboulme-search-agent/1.0 (https://caronboulme.fr)"

AGENT_NAME = "search"
_subscribed_sessions: set[str] = set()

# ---------------------------------------------------------------------------
# MQTT Connection Helper
# ---------------------------------------------------------------------------

async def create_nexus_client() -> NexusClient:
    """Crée un NexusClient connecté via Authentik OAuth."""
    import httpx

    logger.info("Connexion MQTT via Authentik OAuth...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AUTHENTIK_URL}/application/o/token/",
            data={
                "grant_type": "client_credentials",
                "client_id": AUTHENTIK_CLIENT_ID,
                "client_secret": AUTHENTIK_CLIENT_SECRET,
            },
        )
        if resp.status_code != 200:
            logger.error(f"Échec obtention token OAuth: {resp.status_code} {resp.text}")
            raise RuntimeError("Cannot get OAuth token")

        token_data = resp.json()
        access_token = token_data["access_token"]
        logger.info(f"Token OAuth obtenu (expire dans {token_data['expires_in']}s)")

    nexus = await NexusClient.from_authentik_token(
        AUTHENTIK_URL,
        MQTT_HOST,
        access_token,
        AUTHENTIK_CLIENT_ID,
        AUTHENTIK_CLIENT_SECRET,
        MQTT_PORT,
    )
    logger.info(f"NexusClient créé avec username: {nexus._username}")
    return nexus

# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

# La contrainte de longueur de chaque prompt est ce qui borne le temps de
# réponse : le coût d'un appel tient à ce que le modèle génère, pas à ce qu'il
# lit (diviser le contexte par 3,7 ne gagne que 0,7s, alors que passer de 5 à
# 3 phrases en gagne 1,5). Un prompt qui demande d'être « dense » ou
# « exhaustif » produit l'inverse de l'effet recherché : le modèle part en
# listes et double son temps.
DETAIL_SYSTEM_PROMPTS = {
    1: (
        "Réponds à la question en UNE seule phrase directe et naturelle, à l'oral. "
        "Utilise uniquement les informations des résultats de recherche fournis. "
        "Pas de citation de sources."
    ),
    2: (
        "Rédige un paragraphe clair et factuel répondant à la question, "
        "basé uniquement sur les résultats de recherche fournis. "
        "2 à 3 phrases maximum. Français, ton oral. Pas de liste ni de tirets."
    ),
    3: (
        "Rédige une réponse détaillée avec les faits pertinents des résultats de recherche, "
        "en 5 phrases maximum. "
        "Cite les sources naturellement quand c'est pertinent (ex: 'selon Le Monde...', 'd'après Wikipedia...'). "
        "Français, ton oral. Pas de liste ni de tirets."
    ),
}

# Borne haute par niveau, pour qu'un modèle qui ignore la consigne de longueur
# ne fasse pas exploser le temps de réponse. Sans elle, le niveau 3 atteignait
# le plafond de 600 tokens et mettait 19s.
MAX_TOKENS_BY_LEVEL = {1: 120, 2: 250, 3: 450}

# Ajouté au prompt quand les deux volets portent du contenu. Les résultats
# arrivent au LLM sous deux intitulés (FOND et ACTUALITÉ RÉCENTE) : sans cette
# consigne il les mélange et noie la chronologie, alors que l'intérêt de
# chercher les deux est justement de distinguer ce qui est établi de ce qui
# vient de bouger.
STRUCTURE_HINT = (
    " Les résultats sont groupés en deux ensembles. Commence par situer le "
    "sujet à partir du FOND, puis enchaîne sur ce qui a bougé récemment à "
    "partir de l'ACTUALITÉ RÉCENTE, en le signalant naturellement à l'oral "
    "('ces derniers jours...', 'plus récemment...'). Si l'un des deux "
    "ensembles n'apporte rien sur la question, ne le mentionne pas et "
    "réponds avec l'autre."
)

REPORT_TOOL = [{
    "type": "function",
    "function": {
        "name": "report_search",
        "description": "Report the search answer as natural spoken French.",
        "parameters": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "description": "The answer in French, natural spoken language.",
                }
            },
            "required": ["report"],
        },
    },
}]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _format_results(results: list[dict], start: int = 1) -> str:
    return "\n\n".join(
        f"[{start + i}] {r.get('title', '')}\nSource: {r.get('url', '')}\n{r.get('content', '')}"
        for i, r in enumerate(results)
        if r.get("content")
    )


def _synthesize_sync(query: str, background: list[dict], recent: list[dict],
                     detail_level: int) -> str:
    """Synthétise les deux volets en une seule réponse orale.

    Un seul appel LLM plutôt qu'un par volet : le modèle a besoin de voir le
    fond pour savoir ce qui, dans l'actualité, mérite d'être signalé comme
    nouveau.
    """
    system_prompt = DETAIL_SYSTEM_PROMPTS.get(detail_level, DETAIL_SYSTEM_PROMPTS[2])

    bg_text = _format_results(background[:10])
    rc_text = _format_results(recent[:8], start=len(background[:10]) + 1)

    # detail_level 1 attend une phrase unique : la structurer en deux temps
    # produirait exactement ce que ce niveau cherche à éviter.
    if bg_text and rc_text and detail_level > 1:
        system_prompt += STRUCTURE_HINT

    sections = []
    if bg_text:
        sections.append(f"FOND (encyclopédie et sources de référence):\n{bg_text}")
    if rc_text:
        sections.append(f"ACTUALITÉ RÉCENTE (presse):\n{rc_text}")
    results_text = "\n\n".join(sections)

    user_content = f"Question: {query}\n\nRésultats de recherche:\n{results_text}"

    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=REPORT_TOOL,
            tool_choice="required",
            max_tokens=MAX_TOKENS_BY_LEVEL.get(detail_level, 250),
        )
        msg = resp.choices[0].message
        report = ""

        if msg.tool_calls:
            try:
                report = json.loads(msg.tool_calls[0].function.arguments).get("report", "")
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                logger.warning(f"Tool call illisible: {e}")

        if not report and msg.content:
            # vLLM annonce finish_reason=tool_calls mais ne parse aucun tool call
            # avec ce modèle : la synthèse arrive telle quelle dans content. Sans
            # cette reprise, on la jetait pour renvoyer results_text[:500], soit un
            # extrait brut tronqué après plusieurs secondes d'attente.
            content = msg.content.strip()
            if content.startswith("{"):
                try:
                    content = json.loads(content).get("report", content)
                except json.JSONDecodeError:
                    pass
            report = content

        logger.info(f"Synthèse (level={detail_level}): {report[:100]!r}...")
        return report or results_text[:500]
    except Exception as e:
        logger.error(f"Synthèse échouée: {e}")
        return results_text[:500]


# ---------------------------------------------------------------------------
# Filtrage des résultats
# ---------------------------------------------------------------------------

# Sites de dictionnaire/définition/mots-fléchés : toujours assez longs pour
# passer le filtre de qualité par longueur, mais jamais pertinents pour une
# question factuelle — la requête matche le MOT ('vainqueur', 'adversaire'),
# pas le SUJET recherché. À exclure par nom plutôt que par longueur.
_NOISE_DOMAINS = (
    "larousse.fr", "lerobert.com", "cnrtl.fr", "wiktionary.org",
    "fsolver.fr", "wordreference.com", "linternaute.fr/dictionnaire",
    "synonymes.com", "cordial.fr",
)


def _is_reference_noise(url: str) -> bool:
    return any(d in url for d in _NOISE_DOMAINS)


_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SearXNG/1.0; +https://searxng.org)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "fr,en;q=0.9",
}


async def _fetch_page_content(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(
            url,
            headers=_FETCH_HEADERS,
            timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return None
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct:
                return None
            html = await resp.text(errors="replace")
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if text:
                return text[:FETCH_MAX_CHARS]
            return None
    except Exception:
        return None


async def _enrich_results(results: list[dict]) -> list[dict]:
    """Récupère le contenu réel des N premières pages, sinon garde l'extrait.

    Les résultats marqués no_fetch (extraits Wikipédia) sont laissés tels
    quels : leur contenu est déjà le résumé voulu, et fetcher la page
    ramènerait l'article entier avec ses annexes.
    """
    to_fetch = [r for r in results[:FETCH_TOP_N] if r.get("url") and not r.get("no_fetch")]
    async with aiohttp.ClientSession() as session:
        fetched = await asyncio.gather(*[_fetch_page_content(session, r["url"]) for r in to_fetch])
    by_url = dict(zip((r["url"] for r in to_fetch), fetched))

    enriched = []
    for result in results:
        r = dict(result)
        full = by_url.get(r.get("url"))
        if full:
            r["content"] = full
            logger.info(f"Fetch OK: {r['url'][:60]} ({len(full)} chars)")
        enriched.append(r)
    return enriched


def _dedup_results(results: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# ddgs — métamoteur qui imite l'empreinte TLS d'un navigateur (via primp), là où
# SearXNG se fait renvoyer un CAPTCHA par les moteurs web généralistes.
# ---------------------------------------------------------------------------

def _ddgs_search_sync(query: str, n: int, recent: bool) -> list[dict]:
    """Interroge les backends ddgs dans l'ordre, s'arrête au premier qui répond.

    recent bascule sur l'API news de ddgs, dont les résultats sont datés et
    triés par fraîcheur. Le filtre timelimit de l'API text ne remplace pas ce
    basculement : yandex l'ignore et bing continue de remonter des pages
    intemporelles (Wikipédia, TikTok) malgré lui.

    ddgs est synchrone et bloquant : à appeler dans un executor.
    """
    is_news = recent
    backends = DDGS_NEWS_BACKENDS if is_news else DDGS_TEXT_BACKENDS

    for backend in backends:
        backend = backend.strip()
        if not backend:
            continue
        try:
            client = DDGS(timeout=DDGS_TIMEOUT)
            search = client.news if is_news else client.text
            raw = search(query, backend=backend, max_results=n, region=DDGS_REGION)
        except DDGSException as e:
            # "No results found" inclus : un backend bloqué lève ici, on essaie le suivant.
            logger.info(f"ddgs[{backend}] '{query[:50]}': {e}")
            continue
        except Exception as e:
            logger.warning(f"ddgs[{backend}] erreur inattendue: {type(e).__name__}: {e}")
            continue

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                # text() renvoie 'body', news() renvoie 'body' aussi : ce n'est
                # qu'un extrait, _enrich_results ira chercher la page entière.
                "content": (r.get("body") or "")[:FETCH_MAX_CHARS],
            }
            for r in raw
            if not _is_reference_noise(r.get("href") or r.get("url", ""))
        ]
        mode = "news" if is_news else "text"
        if results:
            logger.info(f"ddgs[{backend}/{mode}] '{query[:50]}': {len(results)} résultats")
            return results[:n]

    logger.warning(f"ddgs: aucun backend n'a répondu pour '{query[:50]}' (recent={recent})")
    return []


async def _ddgs_search(query: str, n: int, recent: bool = False) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _ddgs_search_sync, query, n, recent)


# ---------------------------------------------------------------------------
# Wikipédia — volet encyclopédique du fond
# ---------------------------------------------------------------------------

async def _wikipedia_search(query: str, n: int = WIKIPEDIA_N) -> list[dict]:
    """Cherche des articles Wikipédia et récupère leur introduction.

    Deux appels à l'API MediaWiki : list=search pour trouver les articles,
    prop=extracts pour leur résumé. L'introduction suffit et évite le hors-sujet
    des sections annexes ; ces résultats n'ont donc pas besoin d'être enrichis
    par un fetch de page.
    """
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": WIKIPEDIA_UA}) as session:
            async with session.get(
                WIKIPEDIA_API,
                params={
                    "action": "query", "list": "search", "srsearch": query,
                    "srlimit": n, "format": "json",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                hits = (await resp.json()).get("query", {}).get("search", [])

            if not hits:
                logger.info(f"Wikipédia '{query[:50]}': aucun article")
                return []

            async with session.get(
                WIKIPEDIA_API,
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "exintro": 1, "titles": "|".join(h["title"] for h in hits),
                    "format": "json",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                pages = (await resp.json()).get("query", {}).get("pages", {})

            results = [
                {
                    "title": p.get("title", ""),
                    "url": f"https://fr.wikipedia.org/wiki/{quote(p.get('title', '').replace(' ', '_'))}",
                    "content": (p.get("extract") or "")[:FETCH_MAX_CHARS],
                    "no_fetch": True,  # l'extrait fait foi, inutile d'aller chercher la page
                }
                for p in pages.values()
                if p.get("extract")
            ]
            logger.info(f"Wikipédia '{query[:50]}': {len(results)} articles")
            return results
    except Exception as e:
        logger.error(f"Wikipédia échoué: {e}")
        return []


async def _anysearch_search(query: str, n: int) -> list[dict]:
    if not ANYSEARCH_API_KEY:
        return []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ANYSEARCH_URL,
                headers={"Authorization": f"Bearer {ANYSEARCH_API_KEY}"},
                json={"query": query, "max_results": n},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": (r.get("content") or r.get("snippet") or "")[:FETCH_MAX_CHARS],
                    }
                    for r in data.get("data", {}).get("results", [])[:n]
                ]
                logger.info(f"AnySearch '{query}': {len(results)} résultats")
                return results
    except Exception as e:
        logger.error(f"AnySearch échoué: {e}")
        return []


async def _search_two_sided(query: str, n: int, log_prefix: str = "") -> tuple[list[dict], list[dict]]:
    """Cherche le fond et l'actualité en parallèle, renvoie (fond, récent).

    Trois sources lancées ensemble : Wikipédia et le web pour le fond, la
    presse pour l'actualité. Le coût en latence est celui de la plus lente,
    pas de leur somme.

    AnySearch ne prend le relais que si le web ET la presse sont muets :
    Wikipédia seul ne suffit pas à considérer la recherche réussie, il répond
    presque toujours quelque chose, fût-ce à côté.
    """
    wiki, web, press = await asyncio.gather(
        _wikipedia_search(query),
        _ddgs_search(query, n, recent=False),
        _ddgs_search(query, n, recent=True),
    )

    if not web and not press:
        logger.info(f"{log_prefix} ddgs muet des deux côtés — fallback AnySearch")
        web = await _anysearch_search(query, n)

    background_raw = _dedup_results(wiki + web)
    recent_raw = _dedup_results(press)

    background, recent_res = await asyncio.gather(
        _enrich_results(background_raw) if background_raw else _noop_list(),
        _enrich_results(recent_raw) if recent_raw else _noop_list(),
    )
    logger.info(
        f"{log_prefix} fond: {len(background)} sources "
        f"({len(wiki)} wikipédia) — actualité: {len(recent_res)} sources"
    )
    return background, recent_res


async def _noop_list() -> list[dict]:
    return []


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

async def on_user_connected(topic: str, payload):
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not password or not session_id:
        return

    agent_topics_topic = None
    for agent_entry in private_topics:
        for t in agent_entry.get("topics", []):
            if t["topic"].endswith("/agent_topics"):
                agent_topics_topic = t["topic"]
                break

    if not agent_topics_topic:
        logger.warning(f"[{username}] agent_topics topic introuvable, skip")
        return

    request_topic = f"users/{username}/{session_id}/search/request"
    result_topic = f"users/{username}/{session_id}/search/result"

    nexus = await create_nexus_client()

    await nexus.publish(
        agent_topics_topic,
        [{
            "agent": AGENT_NAME,
            "topics": [
                {
                    "topic": request_topic,
                    "description": (
                        "Recherche sur internet — pour toute question factuelle précise "
                        "(sujet, personne, lieu, définition, valeur boursière, événement). "
                        "Poser la question telle quelle : la recherche interroge à la fois "
                        "l'encyclopédie et le web pour le fond du sujet, et la presse pour "
                        "ce qui a bougé récemment, puis rend une réponse qui enchaîne les "
                        "deux. Rien d'autre à préciser."
                    ),
                    "access": "write",
                    "response_topic": result_topic,
                    "format": {
                        "query": "string",
                        "n_results": 8,
                        "detail_level": 2,
                    },
                },
                {
                    "topic": result_topic,
                    "description": (
                        "Résultat de la recherche internet. Utiliser le champ 'report' pour "
                        "répondre : il couvre déjà le fond du sujet puis l'actualité récente. "
                        "'background' et 'recent' donnent les sources brutes de chaque volet, "
                        "pour les agents qui veulent les traiter séparément."
                    ),
                    "access": "read",
                    "format": {
                        "report": "string",
                        "background": "string",
                        "recent": "string",
                        "sources": [{"title": "string", "url": "string"}],
                    },
                },
            ],
        }],
    )
    logger.info(f"[{username}/{session_id}] Topics déclarés sur {agent_topics_topic}")

    if session_id in _subscribed_sessions:
        logger.debug(f"[{username}/{session_id}] Déjà abonné, skip")
        return

    _subscribed_sessions.add(session_id)

    async def on_search_request(t: str, p):
        if not isinstance(p, dict):
            logger.warning(f"[{username}/{session_id}] Payload non-JSON/non-dict reçu sur {t}, ignoré: {p!r}")
            await nexus.publish(result_topic, {
                "report": "",
                "sources": [],
                "error": "malformed request payload: expected a JSON object",
            })
            return

        raw_query = p.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            logger.warning(f"[{username}/{session_id}] Requête sans 'query', ignorée: {p}")
            await nexus.publish(result_topic, {
                "report": "Erreur: la requête de recherche ne contenait pas de terme à chercher.",
                "sources": [],
            })
            return

        try:
            n_results = int(p.get("n_results", DEFAULT_N_RESULTS))
            detail_level = int(p.get("detail_level", 2))
            if detail_level not in (1, 2, 3):
                detail_level = 2
            logger.info(f"[{username}] Recherche: '{query}' n={n_results} level={detail_level}")

            loop = asyncio.get_event_loop()

            # 1. Fond (wikipédia + web) et actualité (presse), en parallèle
            background, recent_res = await _search_two_sided(
                query, n_results, log_prefix=f"[{username}/{session_id}]"
            )

            if not background and not recent_res:
                await nexus.publish(result_topic, {
                    "report": "Je n'ai pas trouvé de résultats pour cette recherche.",
                    "sources": [],
                })
                return

            # 2. Synthèse des deux volets en une seule réponse orale
            report = await loop.run_in_executor(
                None, _synthesize_sync, query, background, recent_res, detail_level
            )

            sources = [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in (background[:3] + recent_res[:3])
            ]
            await nexus.publish(result_topic, {
                "report": report,
                "background": _format_results(background[:10]),
                "recent": _format_results(recent_res[:8]),
                "sources": sources,
            })
            logger.info(f"[{username}/{session_id}] Résultat publié sur {result_topic}")
        except Exception as e:
            logger.exception(f"[{username}/{session_id}] Erreur inattendue lors du traitement de la requête search: {e}")
            await nexus.publish(result_topic, {
                "report": "",
                "sources": [],
                "error": f"internal search error: {e}",
            })

    nexus.subscribe(request_topic, on_search_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {request_topic}")


SERVICE_REQUEST_TOPIC = "service/search/request"
SERVICE_RESULT_TOPIC  = "service/search/result"


async def main():
    nexus = await create_nexus_client()

    async def on_service_search_request(topic: str, payload):
        if not isinstance(payload, dict):
            logger.warning(f"[service] Payload non-JSON/non-dict reçu sur {topic}, ignoré: {payload!r}")
            await nexus.publish(SERVICE_RESULT_TOPIC, {
                "report": "",
                "sources": [],
                "error": "malformed request payload: expected a JSON object",
            })
            return

        reply_to = payload.get("reply_to", SERVICE_RESULT_TOPIC)
        raw_query = payload.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            logger.warning(f"[service] Requête sans 'query', ignorée: {payload}")
            await nexus.publish(reply_to, {"report": "Erreur: la requête de recherche ne contenait pas de terme à chercher.", "sources": []})
            return
        try:
            n_results    = int(payload.get("n_results", DEFAULT_N_RESULTS))
            logger.info(f"[service] Recherche: {query!r}")

            background, recent_res = await _search_two_sided(query, n_results, log_prefix="[service]")
            if not background and not recent_res:
                await nexus.publish(reply_to, {"report": "", "sources": []})
                return

            # Pas de synthèse LLM ici : les agents appelants (bulletin news)
            # refont leur propre passe sur le contenu brut, une synthèse
            # intermédiaire ne ferait que leur retirer de la matière.
            bg = _format_results(background[:10])
            rc = _format_results(recent_res[:8], start=len(background[:10]) + 1)
            report = "\n\n---\n\n".join(
                s for s in (
                    f"FOND\n{bg}" if bg else "",
                    f"ACTUALITÉ RÉCENTE\n{rc}" if rc else "",
                ) if s
            )
            sources = [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in (background[:3] + recent_res[:3])
            ]
            await nexus.publish(reply_to, {
                "report": report,
                "background": bg,
                "recent": rc,
                "sources": sources,
            })
            logger.info(f"[service] Résultat publié ({len(report)} chars)")
        except Exception as e:
            logger.error(f"[service] Erreur recherche {query!r}: {e}")
            await nexus.publish(reply_to, {"report": "", "sources": [], "error": f"internal search error: {e}"})

    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.subscribe(SERVICE_REQUEST_TOPIC, on_service_search_request)
    nexus.start_listening()
    logger.info(
        f"Search service démarré — ddgs text={','.join(DDGS_TEXT_BACKENDS)} "
        f"news={','.join(DDGS_NEWS_BACKENDS)}, fallback AnySearch"
    )
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
