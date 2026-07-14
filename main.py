import asyncio
import json
import logging
import os
import sys

import aiohttp
import openai
import trafilatura
from nexus_client import NexusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VK_URL = os.environ["VK_URL"]
MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
SERVICE_USERNAME = os.environ["MQTT_SERVICE_USERNAME"]
SERVICE_API_KEY = os.environ["MQTT_SERVICE_API_KEY"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
DEFAULT_N_RESULTS = int(os.environ.get("DEFAULT_N_RESULTS", "4"))
FETCH_TOP_N = int(os.environ.get("FETCH_TOP_N", "4"))
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "8"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "3000"))
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
ANYSEARCH_API_KEY = os.environ.get("ANYSEARCH_API_KEY", "")
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"

AGENT_NAME = "search"
_subscribed_sessions: set[str] = set()

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


JUDGE_TOOL = [{
    "type": "function",
    "function": {
        "name": "judge_relevance",
        "description": "State whether the search results clearly and currently answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answered": {
                    "type": "boolean",
                    "description": (
                        "True only if the results give a clear, current, factual answer. "
                        "False if they are off-topic (e.g. dictionary definitions of a word "
                        "in the question), or describe the event as still pending/undecided "
                        "at the time of the question."
                    ),
                }
            },
            "required": ["answered"],
        },
    },
}]


def _judge_sync(query: str, results: list[dict]) -> bool:
    """Appel LLM court : les résultats SearXNG répondent-ils vraiment à la question ?
    Remplace un seuil sur la longueur du contenu, qui ne distingue ni le hors-sujet
    verbeux (pages de dictionnaire) ni le contenu pertinent mais périmé."""
    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\n{(r.get('content') or '')[:500]}"
        for i, r in enumerate(results[:8])
        if r.get("content")
    )
    if not results_text:
        return False
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Tu juges la pertinence de résultats de recherche pour une question donnée."},
                {"role": "user", "content": f"Question: {query}\n\nRésultats:\n{results_text}"},
            ],
            tools=JUDGE_TOOL,
            tool_choice="required",
            max_tokens=50,
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return True  # échec du tool-calling : ne pas bloquer sur une erreur de jugement
        answered = json.loads(tool_calls[0].function.arguments).get("answered", True)
        logger.info(f"Juge: answered={answered}")
        return bool(answered)
    except Exception as e:
        logger.error(f"Jugement échoué: {e}")
        return True  # fail-open : en cas d'erreur, ne pas consommer le quota Tavily pour rien


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
# Tavily (fallback quand SearXNG renvoie peu/pas de contenu exploitable)
# ---------------------------------------------------------------------------

async def _tavily_search(query: str, categories: str, n: int) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    topic = "news" if categories == "news" else "general"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": topic,
        "search_depth": "advanced",
        "max_results": min(n, 10),
        "include_raw_content": True,
    }
    if topic == "news":
        payload["time_range"] = "week"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TAVILY_URL, json=payload, timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": (r.get("raw_content") or r.get("content") or "")[:FETCH_MAX_CHARS],
                    }
                    for r in data.get("results", [])[:n]
                ]
                logger.info(f"Tavily '{query}' ({topic}): {len(results)} résultats")
                return results
    except Exception as e:
        logger.error(f"Tavily échoué: {e}")
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


async def _search_with_fallback(query: str, categories: str, n: int, log_prefix: str = "",
                                 allow_tavily: bool = True) -> list[dict]:
    """SearXNG (multi-moteurs, gratuit) et AnySearch (quota généreux) sont
    toujours interrogés en parallèle, quel que soit le résultat de l'un ou
    l'autre. Un LLM juge ensuite si l'ensemble combiné répond vraiment à la
    question — hors-sujet ou périmé compte comme non-réponse — et Tavily
    (quota rare) prend le relais seulement dans ce cas, sauf si allow_tavily=False
    (appels à fort volume, ex: le deep dive du bulletin news)."""
    raw_results, anysearch_results = await asyncio.gather(
        _search(query, categories, n),
        _anysearch_search(query, n),
    )
    enriched = await _enrich_results(raw_results) if raw_results else []
    combined = _dedup_results(enriched + anysearch_results)

    if not allow_tavily:
        return combined

    loop = asyncio.get_event_loop()
    answered = await loop.run_in_executor(None, _judge_sync, query, combined) if combined else False

    if not answered:
        tavily_results = await _tavily_search(query, categories, n)
        if tavily_results:
            logger.info(f"{log_prefix} SearXNG+AnySearch ne répondent pas à la question — fallback Tavily")
            return tavily_results

    return combined


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

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

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
            return

        query = p.get("query", "").strip()
        if not query:
            logger.warning(f"[{username}/{session_id}] Requête sans 'query', ignorée: {p}")
            await nexus.publish(result_topic, {
                "report": "Erreur: la requête de recherche ne contenait pas de terme à chercher.",
                "sources": [],
            })
            return

        categories = p.get("categories", "general")
        n_results = int(p.get("n_results", DEFAULT_N_RESULTS))
        detail_level = int(p.get("detail_level", 2))
        if detail_level not in (1, 2, 3):
            detail_level = 2
        allow_tavily = bool(p.get("allow_tavily", True))

        logger.info(f"[{username}] Recherche: '{query}' categories={categories} n={n_results} level={detail_level}")

        loop = asyncio.get_event_loop()

        # 1. Search (SearXNG, fallback Tavily si résultats pauvres)
        enriched_results = await _search_with_fallback(
            query, categories, n_results, log_prefix=f"[{username}/{session_id}]", allow_tavily=allow_tavily
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

    nexus.subscribe(request_topic, on_search_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {request_topic}")


SERVICE_REQUEST_TOPIC = "service/search/request"
SERVICE_RESULT_TOPIC  = "service/search/result"


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    async def on_service_search_request(topic: str, payload):
        if not isinstance(payload, dict):
            return
        query = payload.get("query", "").strip()
        reply_to     = payload.get("reply_to", SERVICE_RESULT_TOPIC)
        if not query:
            logger.warning(f"[service] Requête sans 'query', ignorée: {payload}")
            await nexus.publish(reply_to, {"report": "Erreur: la requête de recherche ne contenait pas de terme à chercher.", "sources": []})
            return
        categories   = payload.get("categories", "general")
        n_results    = int(payload.get("n_results", DEFAULT_N_RESULTS))
        detail_level = int(payload.get("detail_level", 2))
        if detail_level not in (1, 2, 3):
            detail_level = 2
        allow_tavily = bool(payload.get("allow_tavily", True))
        logger.info(f"[service] Recherche: {query!r} categories={categories}")
        try:
            enriched = await _search_with_fallback(query, categories, n_results, log_prefix="[service]", allow_tavily=allow_tavily)
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
            await nexus.publish(reply_to, {"report": "", "sources": []})

    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.subscribe(SERVICE_REQUEST_TOPIC, on_service_search_request)
    nexus.start_listening()
    logger.info(f"Search service démarré — SearXNG: {SEARXNG_URL}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
