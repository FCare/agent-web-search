import asyncio
import json
import logging
import os
import sys

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
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
DEFAULT_N_RESULTS = int(os.environ.get("DEFAULT_N_RESULTS", "4"))
FETCH_TOP_N = int(os.environ.get("FETCH_TOP_N", "4"))
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "8"))
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

# Catégories servies par ddgs plutôt que par SearXNG : ce sont celles où les
# moteurs web généralistes de SearXNG se font bloquer (CAPTCHA/429). Les autres
# (science, it, music, videos, images, social media, map) restent sur SearXNG,
# qui y agrège des sources spécialisées que ddgs n'a pas (pubmed, arxiv, github,
# stackoverflow, mastodon...).
DDGS_CATEGORIES = {"general", "news"}

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

DETAIL_SYSTEM_PROMPTS = {
    1: (
        "Réponds à la question en UNE seule phrase directe et naturelle, à l'oral. "
        "Utilise uniquement les informations des résultats de recherche fournis. "
        "Pas de citation de sources."
    ),
    2: (
        "Rédige un paragraphe clair et factuel répondant à la question, "
        "basé uniquement sur les résultats de recherche fournis. "
        "3 à 5 phrases. Français, ton oral. Pas de liste ni de tirets."
    ),
    3: (
        "Rédige une réponse détaillée avec tous les faits pertinents des résultats de recherche. "
        "Cite les sources naturellement quand c'est pertinent (ex: 'selon Le Monde...', 'd'après Wikipedia...'). "
        "Français, ton oral. Pas de liste ni de tirets."
    ),
}

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

def _synthesize_sync(query: str, results: list[dict], detail_level: int) -> str:
    system_prompt = DETAIL_SYSTEM_PROMPTS.get(detail_level, DETAIL_SYSTEM_PROMPTS[2])

    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\nSource: {r.get('url', '')}\n{r.get('content', '')}"
        for i, r in enumerate(results[:15])
        if r.get("content")
    )
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
            max_tokens=600,
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return results_text[:500]
        report = json.loads(tool_calls[0].function.arguments).get("report", "")
        logger.info(f"Synthèse (level={detail_level}): {report[:100]!r}...")
        return report or results_text[:500]
    except Exception as e:
        logger.error(f"Synthèse échouée: {e}")
        return results_text[:500]


# ---------------------------------------------------------------------------
# SearXNG
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


async def _search(query: str, categories: str, n: int) -> list[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": categories,
                    "language": "fr-FR",
                    "pageno": 1,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                all_results = [r for r in data.get("results", []) if not _is_reference_noise(r.get("url", ""))]
                results = all_results[:n]
                logger.info(f"SearXNG '{query}' ({categories}): {len(results)} résultats")
                return results
    except Exception as e:
        logger.error(f"SearXNG échoué: {e}")
        return []


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
    """Fetch full page content for top N results, fallback to SearXNG snippet."""
    to_fetch = [r for r in results[:FETCH_TOP_N] if r.get("url")]
    async with aiohttp.ClientSession() as session:
        fetched = await asyncio.gather(*[_fetch_page_content(session, r["url"]) for r in to_fetch])

    enriched = []
    fetch_idx = 0
    for i, result in enumerate(results):
        r = dict(result)
        if i < FETCH_TOP_N:
            full = fetched[fetch_idx]
            fetch_idx += 1
            if full:
                r["content"] = full
                logger.info(f"Fetch OK: {r['url'][:60]} ({len(full)} chars)")
            else:
                logger.debug(f"Fetch échoué, fallback snippet: {r.get('url', '')[:60]}")
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

def _ddgs_search_sync(query: str, categories: str, n: int) -> list[dict]:
    """Interroge les backends ddgs dans l'ordre, s'arrête au premier qui répond.

    ddgs est synchrone et bloquant : à appeler dans un executor.
    """
    is_news = categories == "news"
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
        if results:
            logger.info(f"ddgs[{backend}] '{query[:50]}' ({categories}): {len(results)} résultats")
            return results[:n]

    logger.warning(f"ddgs: aucun backend n'a répondu pour '{query[:50]}' ({categories})")
    return []


async def _ddgs_search(query: str, categories: str, n: int) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _ddgs_search_sync, query, categories, n)


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


async def _search_with_fallback(query: str, categories: str, n: int, log_prefix: str = "") -> list[dict]:
    """Route la requête vers le moteur adapté à la catégorie, puis enrichit.

    general et news vont chez ddgs, qui passe là où les moteurs web de SearXNG
    se font bloquer ; AnySearch prend le relais si aucun backend ddgs ne répond.
    Les autres catégories vont chez SearXNG, seul à agréger les sources
    spécialisées correspondantes (pubmed, arxiv, github, mastodon...).

    Dans les deux cas les résultats ne portent qu'un extrait : _enrich_results
    va chercher le contenu réel des pages.
    """
    if categories in DDGS_CATEGORIES:
        results = await _ddgs_search(query, categories, n)
        if not results:
            logger.info(f"{log_prefix} ddgs muet — fallback AnySearch")
            results = await _anysearch_search(query, n)
    else:
        results = await _search(query, categories, n)

    if not results:
        return []
    return _dedup_results(await _enrich_results(results))


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
                        "Recherche sur internet via SearXNG — pour toute question factuelle "
                        "précise (sujet, personne, lieu, définition, valeur boursière, événement). "
                        "Catégories: general, news, science, it, social+media, map, music, videos, images."
                    ),
                    "access": "write",
                    "response_topic": result_topic,
                    "format": {
                        "query": "string",
                        "categories": "general",
                        "n_results": 8,
                        "detail_level": 2,
                    },
                },
                {
                    "topic": result_topic,
                    "description": "Résultat de la recherche internet. Utiliser le champ 'report' pour répondre.",
                    "access": "read",
                    "format": {
                        "report": "string",
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
            categories = p.get("categories", "general")
            n_results = int(p.get("n_results", DEFAULT_N_RESULTS))
            detail_level = int(p.get("detail_level", 2))
            if detail_level not in (1, 2, 3):
                detail_level = 2
            logger.info(f"[{username}] Recherche: '{query}' categories={categories} n={n_results} level={detail_level}")

            loop = asyncio.get_event_loop()

            # 1. Search (ddgs ou SearXNG selon la catégorie, fallback AnySearch)
            enriched_results = await _search_with_fallback(
                query, categories, n_results, log_prefix=f"[{username}/{session_id}]"
            )

            if not enriched_results:
                await nexus.publish(result_topic, {
                    "report": "Je n'ai pas trouvé de résultats pour cette recherche.",
                    "sources": [],
                })
                return

            # 2. Synthesize from enriched content
            report = await loop.run_in_executor(None, _synthesize_sync, query, enriched_results, detail_level)

            sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in enriched_results[:5]]
            await nexus.publish(result_topic, {
                "report": report,
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
            categories   = payload.get("categories", "general")
            n_results    = int(payload.get("n_results", DEFAULT_N_RESULTS))
            detail_level = int(payload.get("detail_level", 2))
            if detail_level not in (1, 2, 3):
                detail_level = 2
            logger.info(f"[service] Recherche: {query!r} categories={categories}")

            enriched = await _search_with_fallback(query, categories, n_results, log_prefix="[service]")
            if not enriched:
                await nexus.publish(reply_to, {"report": "", "sources": []})
                return
            parts = [
                f"[{r.get('title', '')}]\nSource: {r.get('url', '')}\n{(r.get('content') or '')[:1500]}"
                for r in enriched if r.get("content") or r.get("title")
            ]
            report = "\n\n---\n\n".join(parts)
            sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in enriched[:5]]
            await nexus.publish(reply_to, {"report": report, "sources": sources})
            logger.info(f"[service] Résultat publié ({len(report)} chars)")
        except Exception as e:
            logger.error(f"[service] Erreur recherche {query!r}: {e}")
            await nexus.publish(reply_to, {"report": "", "sources": [], "error": f"internal search error: {e}"})

    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.subscribe(SERVICE_REQUEST_TOPIC, on_service_search_request)
    nexus.start_listening()
    logger.info(f"Search service démarré — SearXNG: {SEARXNG_URL}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
