import asyncio
import json
import logging
import os
import sys

from urllib.parse import quote

import aiohttp
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
DEFAULT_N_RESULTS = int(os.environ.get("DEFAULT_N_RESULTS", "4"))
FETCH_TOP_N = int(os.environ.get("FETCH_TOP_N", "4"))
# 3s plutôt que 8 : le fetch est parallèle, son coût est celui de la page la
# plus lente. Mesuré sur un lot de 8 pages, sept répondent en moins de 2s et
# une traîne — la borne basse coupe cette traînarde, dont on garde l'extrait du
# moteur, au lieu de faire attendre tout le monde.
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "3"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "3000"))
# Longueur retenue par source dans la réponse, et nombre de sources par volet.
# L'agent ne rédige pas : il rend la matière brute et c'est l'assistant qui
# formule. Ces bornes tiennent l'ensemble autour de 9 000 caractères, le point
# où son tour de parole reste rapide.
RESULT_MAX_CHARS = int(os.environ.get("RESULT_MAX_CHARS", "800"))
BACKGROUND_N = int(os.environ.get("BACKGROUND_N", "6"))
RECENT_N = int(os.environ.get("RECENT_N", "5"))
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

# UA descriptif du service, envoyé sur tout appel HTTP sortant (fetch de page
# comme API Wikipédia) — identifie l'agent au lieu de se faire passer pour un
# autre projet (ex: SearXNG, qu'on n'utilise plus).
SERVICE_UA = "caronboulme-search-agent/1.0 (https://caronboulme.fr)"

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
    "User-Agent": SERVICE_UA,
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


def _format_results(results: list[dict], start: int = 1) -> str:
    """Met les résultats en forme pour l'appelant, tronqués par source.

    La troncature n'est pas cosmétique, elle décide du temps de réponse de
    l'assistant : mesuré sur un tour de parole complet, 9 000 caractères lui
    coûtent 3,1s contre 9,0s pour 36 000. Au-delà, il paie le prefill sans rien
    y gagner — les premières lignes d'un article portent l'essentiel.
    """
    return "\n\n".join(
        f"[{start + i}] {r.get('title', '')}\nSource: {r.get('url', '')}\n"
        f"{(r.get('content') or '')[:RESULT_MAX_CHARS]}"
        for i, r in enumerate(results)
        if r.get("content")
    )


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
        async with aiohttp.ClientSession(headers={"User-Agent": SERVICE_UA}) as session:
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


def _fit_to_budget(text: str, budget: int) -> str:
    """Coupe à la fin du dernier résultat qui tient dans le budget.

    Couper au milieu d'une source donnerait une phrase tronquée que le LLM
    lecteur prendrait pour un fait complet ; mieux vaut rendre une source de
    moins et que chacune soit entière.
    """
    if len(text) <= budget:
        return text
    blocs = text.split("\n\n")
    gardes, taille = [], 0
    for bloc in blocs:
        if taille + len(bloc) + 2 > budget:
            break
        gardes.append(bloc)
        taille += len(bloc) + 2
    # Aucun bloc entier ne tient : on rend le premier tronqué plutôt que rien.
    return "\n\n".join(gardes) if gardes else blocs[0][:budget]


def _build_response(background: list[dict], recent: list[dict],
                    max_chars: int | None = None) -> dict:
    """Assemble la réponse rendue à l'appelant.

    L'agent ne rédige pas : il rend la matière, l'appelant formule. Mesuré sur
    un tour de parole complet, faire synthétiser l'agent coûtait 11,1s pour un
    résultat moins précis que 3,1s en rendant le brut — la double reformulation
    perdait les dates et les chiffres au passage.

    max_chars laisse l'appelant fixer son budget de contexte. L'agent le
    répartit lui-même à parts égales entre les deux volets et coupe sur une
    frontière de source : un appelant qui tronquerait la réponse concaténée
    perdrait l'actualité, qui vient en second.

    Les deux volets sont ordonnés par pertinence décroissante — Wikipédia
    d'abord pour le fond —, donc ce qui saute en cas de budget serré est
    toujours le moins pertinent.
    """
    bg = _format_results(background[:BACKGROUND_N])
    rc = _format_results(recent[:RECENT_N], start=len(background[:BACKGROUND_N]) + 1)

    if max_chars:
        # Moitié pour chacun, puis on repasse au second ce que le premier n'a
        # pas consommé : couper sur une frontière de source laisse souvent un
        # reliquat, autant qu'il serve.
        moitie = max(max_chars // 2, 200)
        bg = _fit_to_budget(bg, moitie)
        rc = _fit_to_budget(rc, max(max_chars - len(bg), 200))

    report = "\n\n---\n\n".join(
        s for s in (f"FOND\n{bg}" if bg else "", f"ACTUALITÉ RÉCENTE\n{rc}" if rc else "") if s
    )
    return {
        "report": report,
        "background": bg,
        "recent": rc,
        "sources": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in (background[:3] + recent[:3])
        ],
    }


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
                        "ce qui a bougé récemment. Rien d'autre à préciser."
                    ),
                    "access": "write",
                    "response_topic": result_topic,
                    "format": {
                        "query": "string",
                        "n_results": 8,
                        "max_chars": 0,
                    },
                },
                {
                    "topic": result_topic,
                    "description": (
                        "Extraits des sources trouvées, à reformuler pour répondre. Le champ "
                        "'report' contient les deux volets à la suite : FOND (encyclopédie et "
                        "sources de référence) puis ACTUALITÉ RÉCENTE (presse). 'background' et "
                        "'recent' donnent chaque volet séparément. Ce sont des extraits bruts, "
                        "pas une réponse rédigée : les reformuler, ne jamais les lire tels quels."
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
            max_chars = int(p.get("max_chars", 0)) or None
            logger.info(f"[{username}] Recherche: '{query}' n={n_results} max_chars={max_chars}")

            background, recent_res = await _search_two_sided(
                query, n_results, log_prefix=f"[{username}/{session_id}]"
            )

            if not background and not recent_res:
                await nexus.publish(result_topic, {
                    "report": "Je n'ai pas trouvé de résultats pour cette recherche.",
                    "sources": [],
                })
                return

            await nexus.publish(result_topic, _build_response(background, recent_res, max_chars))
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
            max_chars    = int(payload.get("max_chars", 0)) or None
            logger.info(f"[service] Recherche: {query!r} max_chars={max_chars}")

            background, recent_res = await _search_two_sided(query, n_results, log_prefix="[service]")
            if not background and not recent_res:
                await nexus.publish(reply_to, {"report": "", "sources": []})
                return

            response = _build_response(background, recent_res, max_chars)
            await nexus.publish(reply_to, response)
            logger.info(f"[service] Résultat publié ({len(response['report'])} chars)")
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
