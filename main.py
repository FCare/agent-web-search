import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

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
CONTEXT_TTL_SECONDS = int(os.environ.get("CONTEXT_TTL_SECONDS", str(2 * 3600)))
CONTEXT_MAX_ENTRIES = int(os.environ.get("CONTEXT_MAX_ENTRIES", "10"))
FETCH_TOP_N = int(os.environ.get("FETCH_TOP_N", "4"))
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "8"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "3000"))

AGENT_NAME = "search"
_subscribed_users: set[str] = set()

# ---------------------------------------------------------------------------
# User search context
# ---------------------------------------------------------------------------

@dataclass
class SearchEntry:
    query: str
    topic: str
    results: list[dict]
    report: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UserSearchContext:
    def __init__(self):
        self._entries: list[SearchEntry] = []

    def recent(self) -> list[SearchEntry]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=CONTEXT_TTL_SECONDS)
        self._entries = [e for e in self._entries if e.timestamp > cutoff]
        return self._entries[-CONTEXT_MAX_ENTRIES:]

    def add(self, entry: SearchEntry):
        self._entries.append(entry)


_user_contexts: dict[str, UserSearchContext] = {}


def _get_context(username: str) -> UserSearchContext:
    if username not in _user_contexts:
        _user_contexts[username] = UserSearchContext()
    return _user_contexts[username]


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------

CLASSIFY_TOOL = [{
    "type": "function",
    "function": {
        "name": "classify_query",
        "description": "Classify the query: is it a follow-up/deepening of a recent topic?",
        "parameters": {
            "type": "object",
            "properties": {
                "is_followup": {
                    "type": "boolean",
                    "description": "True if this query deepens or extends a recent search topic.",
                },
                "topic_index": {
                    "type": "integer",
                    "description": "Index of the related recent topic (0-based). -1 if not a follow-up.",
                },
                "refined_query": {
                    "type": "string",
                    "description": (
                        "If follow-up: a focused search query that combines the new angle with the topic context. "
                        "If new topic: the original query, possibly rephrased for better search results."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": "Short label for this search topic (e.g. 'football ligue 1', 'réchauffement climatique').",
                },
            },
            "required": ["is_followup", "topic_index", "refined_query", "topic"],
        },
    },
}]

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

def _classify_query_sync(query: str, recent: list[SearchEntry]) -> dict:
    if not recent:
        return {"is_followup": False, "topic_index": -1, "refined_query": query, "topic": query[:60]}

    topics_text = "\n".join(
        f"[{i}] sujet='{e.topic}' requête='{e.query}'"
        for i, e in enumerate(recent)
    )
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You analyze a new search query and a list of recent search topics. "
                    "Determine if the new query is a follow-up or deepening of one of the recent topics. "
                    "A follow-up asks for more detail, a different angle, or a related aspect of the same subject. "
                    "If it is a follow-up, generate a focused refined_query that targets the new angle within the topic context. "
                    "Always provide a short topic label. "
                    "Call classify_query."
                )},
                {"role": "user", "content": f"New query: {query}\n\nRecent topics:\n{topics_text}"},
            ],
            tools=CLASSIFY_TOOL,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return {"is_followup": False, "topic_index": -1, "refined_query": query, "topic": query[:60]}
        result = json.loads(tool_calls[0].function.arguments)
        logger.info(f"Classification requête: is_followup={result.get('is_followup')}, topic='{result.get('topic')}', refined='{result.get('refined_query')}'")
        return result
    except Exception as e:
        logger.error(f"Classification échouée: {e}")
        return {"is_followup": False, "topic_index": -1, "refined_query": query, "topic": query[:60]}


def _synthesize_sync(query: str, results: list[dict], detail_level: int, context_report: str | None = None) -> str:
    system_prompt = DETAIL_SYSTEM_PROMPTS.get(detail_level, DETAIL_SYSTEM_PROMPTS[2])

    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\nSource: {r.get('url', '')}\n{r.get('content', '')}"
        for i, r in enumerate(results[:15])
        if r.get("content")
    )
    user_content = f"Question: {query}\n\nRésultats de recherche:\n{results_text}"
    if context_report:
        user_content = f"Contexte de la recherche précédente: {context_report}\n\n{user_content}"

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
                results = data.get("results", [])[:n]
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


async def _enrich_entry_background(entry: SearchEntry, results: list[dict], username: str):
    """Fetch full page content in background and update the context entry silently."""
    try:
        enriched = await _enrich_results(results)
        entry.results = enriched
        logger.info(f"[{username}] Enrichissement background terminé pour '{entry.topic}' ({len(enriched)} résultats)")
    except Exception as e:
        logger.warning(f"[{username}] Enrichissement background échoué: {e}")


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

async def on_user_connected(topic: str, payload):
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    private_topics = payload.get("private_topics", [])

    if not username or not password:
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

    request_topic = f"users/{username}/search/request"
    result_topic = f"users/{username}/search/result"

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    await nexus.publish(
        agent_topics_topic,
        [{
            "agent": AGENT_NAME,
            "topics": [
                {
                    "topic": request_topic,
                    "description": (
                        "Recherche sur internet via SearXNG. "
                        "Utiliser pour toute question factuelle sur un sujet précis, une personne, un lieu, une définition, une valeur boursière ou un événement ciblé. "
                        "Catégories disponibles: general, news, science, it, social+media, map, music, videos, images."
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
                        "topic": "string",
                        "is_followup": "bool",
                        "sources": [{"title": "string", "url": "string"}],
                    },
                },
            ],
        }],
    )
    logger.info(f"[{username}] Topics déclarés sur {agent_topics_topic}")

    if username in _subscribed_users:
        logger.debug(f"[{username}] Déjà abonné, skip")
        return

    _subscribed_users.add(username)

    async def on_search_request(t: str, p):
        if not isinstance(p, dict):
            return

        query = p.get("query", "").strip()
        if not query:
            return

        categories = p.get("categories", "general")
        n_results = int(p.get("n_results", DEFAULT_N_RESULTS))
        detail_level = int(p.get("detail_level", 2))
        if detail_level not in (1, 2, 3):
            detail_level = 2

        logger.info(f"[{username}] Recherche: '{query}' categories={categories} n={n_results} level={detail_level}")

        ctx = _get_context(username)
        recent = ctx.recent()

        loop = asyncio.get_event_loop()

        # 1. Classify + search in parallel (skip classify if no context)
        if recent:
            classification, raw_results = await asyncio.gather(
                loop.run_in_executor(None, _classify_query_sync, query, recent),
                _search(query, categories, n_results),
            )
        else:
            classification = {"is_followup": False, "topic_index": -1, "refined_query": query, "topic": query[:60]}
            raw_results = await _search(query, categories, n_results)

        is_followup = classification.get("is_followup", False)
        topic_index = classification.get("topic_index", -1)
        refined_query = classification.get("refined_query", query)
        topic = classification.get("topic", query[:60])

        # If follow-up with a different refined query, search again with the refined query
        if is_followup and refined_query != query:
            new_results = await _search(refined_query, categories, n_results)
        else:
            new_results = raw_results

        # 2. Merge with context if follow-up
        context_report = None
        all_results = new_results
        if is_followup and 0 <= topic_index < len(recent):
            parent = recent[topic_index]
            context_report = parent.report
            all_results = _dedup_results(parent.results + new_results)
            logger.info(f"[{username}] Follow-up de '{parent.topic}': {len(parent.results)} résultats contexte + {len(new_results)} nouveaux = {len(all_results)} total")

        if not all_results:
            await nexus.publish(result_topic, {
                "report": "Je n'ai pas trouvé de résultats pour cette recherche.",
                "topic": topic,
                "is_followup": is_followup,
                "sources": [],
            })
            return

        # 3. Synthesize immediately from snippets → fast first response
        report = await loop.run_in_executor(None, _synthesize_sync, query, all_results, detail_level, context_report)

        # 4. Store entry with snippet-only results, publish response immediately
        entry = SearchEntry(query=query, topic=topic, results=all_results, report=report)
        ctx.add(entry)

        sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in all_results[:5]]
        await nexus.publish(result_topic, {
            "report": report,
            "topic": topic,
            "is_followup": is_followup,
            "sources": sources,
        })
        logger.info(f"[{username}] Résultat publié sur {result_topic}")

        # 5. Background: fetch full page content and enrich context for future follow-ups
        asyncio.create_task(_enrich_entry_background(entry, new_results, username))

    nexus.subscribe(request_topic, on_search_request)
    nexus.start_listening()
    logger.info(f"[{username}] Abonné à {request_topic}")


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info(f"Search service démarré — SearXNG: {SEARXNG_URL}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
