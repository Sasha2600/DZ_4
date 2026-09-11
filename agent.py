"""DZ_4: Память агента — векторный поиск (Qdrant) + графовые связи.

Агент, который:
  1. Хранит узлы знаний в Qdrant (векторная БД, Docker).
  2. Хранит связи между узлами в виде графа (adjacency list).
  3. Выполняет гибридный поиск: семантический (Qdrant) + структурный (граф).
  4. Генерирует ответ LLM с контекстом из обоих источников.

Запуск:
    .venv/bin/python agent.py                # интерактив
    .venv/bin/python agent.py "вопрос"       # один вопрос
    .venv/bin/python agent.py --demo         # демо-запросы
    .venv/bin/python agent.py --selftest     # самодиагностика без LLM / Qdrant
    .venv/bin/python agent.py --explain      # показать детали поиска
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import openai
from dotenv import load_dotenv

# --- Конфигурация -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "lm-studio"
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-12b-qat")
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dz4_memory")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

KNOWLEDGE_FILE = os.getenv("KNOWLEDGE_FILE", "memory/knowledge.json")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
GRAPH_MAX_HOP = int(os.getenv("GRAPH_MAX_HOP", "2"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR")

logger = logging.getLogger("dz4.agent")


def _resolve(path: str) -> str:
    """Относительный путь — от корня проекта."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# --- Исключения -------------------------------------------------------------

class MemoryError(Exception):
    """Ошибка работы с памятью (Qdrant, граф, эмбеддинг)."""


# --- Модели данных ----------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """Узел знания."""
    id: str
    label: str
    type: str
    text: str


@dataclass(frozen=True)
class Edge:
    """Связь между узлами."""
    from_id: str
    to_id: str
    relation: str


@dataclass
class HybridResult:
    """Результат гибридного поиска."""
    node: Node
    vector_score: float
    graph_score: float
    combined_score: float
    path: str = ""  # описание пути в графе


# --- Графовая память --------------------------------------------------------

class GraphMemory:
    """Хранит узлы и связи, обеспечивает обход графа."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.outgoing.setdefault(node.id, [])
        self.incoming.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        self.outgoing.setdefault(edge.from_id, []).append(edge)
        self.incoming.setdefault(edge.to_id, []).append(edge)

    def neighbors(self, node_id: str, max_hop: int = 1) -> dict[str, int]:
        """BFS от node_id до глубины max_hop. Возвращает {id: distance}."""
        if node_id not in self.nodes:
            return {}
        visited: dict[str, int] = {node_id: 0}
        frontier = [node_id]
        for hop in range(1, max_hop + 1):
            next_frontier: list[str] = []
            for nid in frontier:
                for edge in self.outgoing.get(nid, []):
                    if edge.to_id not in visited:
                        visited[edge.to_id] = hop
                        next_frontier.append(edge.to_id)
                for edge in self.incoming.get(nid, []):
                    if edge.from_id not in visited:
                        visited[edge.from_id] = hop
                        next_frontier.append(edge.from_id)
            frontier = next_frontier
            if not frontier:
                break
        visited.pop(node_id, None)  # сам себя не возвращаем
        return visited

    def get_path(self, from_id: str, to_id: str) -> str:
        """Возвращает строковое описание кратчайшего пути."""
        if from_id == to_id:
            return self.nodes.get(from_id, Node(from_id, from_id, "?", "")).label
        queue = [(from_id, [from_id])]
        visited = {from_id}
        while queue:
            current, path = queue.pop(0)
            for edge in self.outgoing.get(current, []):
                if edge.to_id == to_id:
                    full_path = path + [to_id]
                    labels = [self.nodes[nid].label for nid in full_path if nid in self.nodes]
                    return " → ".join(labels)
                if edge.to_id not in visited:
                    visited.add(edge.to_id)
                    queue.append((edge.to_id, path + [edge.to_id]))
            for edge in self.incoming.get(current, []):
                if edge.from_id == to_id:
                    full_path = path + [to_id]
                    labels = [self.nodes[nid].label for nid in full_path if nid in self.nodes]
                    return " → ".join(labels)
                if edge.from_id not in visited:
                    visited.add(edge.from_id)
                    queue.append((edge.from_id, path + [edge.from_id]))
        return ""


def load_knowledge(path: str) -> GraphMemory:
    """Загружает knowledge.json и строит граф."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = GraphMemory()
    for nd in data["nodes"]:
        graph.add_node(Node(nd["id"], nd["label"], nd["type"], nd["text"]))
    for eg in data["edges"]:
        graph.add_edge(Edge(eg["from"], eg["to"], eg["relation"]))
    return graph


# --- Эмбеддинг --------------------------------------------------------------

def embed_with_llm(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Получает векторы эмбеддингов через LLM-сервер."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [[float(v) for v in d.embedding] for d in resp.data]


def embed_single(client: openai.OpenAI, text: str) -> list[float]:
    """Эмбеддинг одного текста."""
    return embed_with_llm(client, [text])[0]


# --- TF-IDF эмбеддинг (для selftest, без LLM) --------------------------------

_STOP_WORDS = frozenset({
    "и", "в", "во", "на", "по", "с", "со", "из", "за", "от", "о", "об", "а", "но",
    "или", "же", "бы", "не", "да", "что", "как", "кто", "где", "когда", "почему",
    "зачем", "куда", "откуда", "можно", "нужно", "надо", "для", "до", "мой", "моя",
    "какой", "какая", "можно", "это", "этот", "эта", "такой", "такая", "такое",
    "про", "из", "через", "связано", "использует",
    "the", "a", "an", "is", "are", "was", "of", "in", "on", "at", "for", "and",
    "or", "how", "what", "which", "who", "can", "i", "you", "my", "your",
})


def _tokenize(text: str) -> list[str]:
    """Токенизация: нижний регистр, только слова, без стоп-слов, len >= 2."""
    tokens = re.split(r"[^\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) >= 2]


class TfidfEmbedder:
    """Детерминированный TF-IDF «эмбеддер» для selftest.

    Строит IDF по корпусу, затем выдаёт TF-IDF векторы по общему словарю.
    """

    def __init__(self, corpus: list[str], dim: int = 512):
        self.dim = dim
        n_docs = len(corpus) or 1
        # Словарь: топ-dim слов по document frequency
        doc_freq: Counter = Counter()
        for text in corpus:
            unique = set(_tokenize(text))
            for t in unique:
                doc_freq[t] += 1
        # Сортируем по частоте, берём топ-dim
        self.vocabulary = [w for w, _ in doc_freq.most_common(dim)]
        self.idf: dict[str, float] = {}
        for w in self.vocabulary:
            df = doc_freq.get(w, 0) or 1
            self.idf[w] = math.log((n_docs + 1) / (df + 1)) + 1
        self._idf_vec = [self.idf.get(w, 1.0) for w in self.vocabulary]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Возвращает список TF-IDF векторов."""
        return [self._single(text) for text in texts]

    def _single(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        freq: Counter = Counter(tokens)
        total = len(tokens) or 1
        vec = []
        for w in self.vocabulary:
            tf = freq.get(w, 0) / total
            vec.append(tf * self.idf.get(w, 1.0))
        # L2 нормализация
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _make_tfidf_embedder(graph: GraphMemory) -> TfidfEmbedder:
    """Создаёт TF-IDF эмбеддер из узлов графа."""
    corpus = [f"{n.label}: {n.text}" for n in graph.nodes.values()]
    return TfidfEmbedder(corpus)


def _tfidf_embed_wrapper(embedder: TfidfEmbedder) -> Callable[[list[str]], list[list[float]]]:
    """Обёртка embedder.embed для использования в MockQdrantMemory."""
    return embedder.embed


def _random_embed(text: str, dim: int = 1024) -> list[float]:
    """Простой детерминированный вектор — для фолбэка без LLM."""
    import random as _r
    h = sum(ord(c) * (i + 1) for i, c in enumerate(text)) & 0xFFFFFFFF
    rng = _r.Random(h)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# --- Qdrant память ----------------------------------------------------------

class QdrantMemory:
    """Векторная память на Qdrant."""

    def __init__(self, url: str, collection: str, embed_fn: Callable[[str], list[float]]):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embed_fn = embed_fn
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию, если её нет."""
        from qdrant_client.http.models import Distance, VectorParams
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert_nodes(self, nodes: list[Node]) -> None:
        """Вставляет узлы в коллекцию с эмбеддингом."""
        texts = [f"{n.label}: {n.text}" for n in nodes]
        vectors = self.embed_fn(texts)
        from qdrant_client.models import PointStruct
        points = []
        for i, (node, vec) in enumerate(zip(nodes, vectors)):
            points.append(PointStruct(
                id=i,
                vector=vec,
                payload={
                    "node_id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "text": node.text,
                },
            ))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        """Ищет похожие узлы. Возвращает [(node_id, score, payload), ...]."""
        vec = self.embed_fn([query])[0]
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
        )
        return [(p.payload["node_id"], p.score, p.payload) for p in resp.points]

    def close(self) -> None:
        self.client.close()


class MockQdrantMemory:
    """Заглушка Qdrant для selftest — работает без сети."""

    def __init__(self, embed_fn: Callable[[str], list[float]]):
        self.embed_fn = embed_fn
        self.points: list[dict] = []

    def upsert_nodes(self, nodes: list[Node]) -> None:
        texts = [f"{n.label}: {n.text}" for n in nodes]
        vectors = self.embed_fn(texts)
        for node, vec in zip(nodes, vectors):
            self.points.append({
                "id": node.id,
                "vector": vec,
                "payload": {
                    "node_id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "text": node.text,
                },
            })

    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        qvec = self.embed_fn([query])[0]
        scored = []
        for p in self.points:
            sim = cosine_similarity(qvec, p["vector"])
            scored.append((p["payload"]["node_id"], sim, p["payload"]))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def close(self) -> None:
        pass


# --- Гибридный поиск --------------------------------------------------------

def hybrid_search(
    query: str,
    graph: GraphMemory,
    qdrant_mem,
    top_k: int,
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
    max_hop: int = 2,
) -> list[HybridResult]:
    """Гибридный поиск: векторный (Qdrant) + графовый (связи)."""
    # 1. Векторный поиск
    vector_results = qdrant_mem.search(query, top_k=max(top_k, len(graph.nodes)))
    vec_map: dict[str, float] = {rid: score for rid, score, _ in vector_results}

    # 2. Собираем графовых соседей топ-3 векторных хитов
    graph_scores: dict[str, float] = {}
    graph_paths: dict[str, str] = {}
    top_vec_ids = [rid for rid, _, _ in vector_results[:min(3, len(vector_results))]]
    for vid in top_vec_ids:
        neighbors = graph.neighbors(vid, max_hop=max_hop)
        for nid, dist in neighbors.items():
            score = 1.0 / dist  # чем ближе, тем выше
            if nid not in graph_scores or score > graph_scores[nid]:
                graph_scores[nid] = score
                if nid in graph.nodes:
                    graph_paths[nid] = graph.get_path(vid, nid)

    # 3. Объединяем кандидатов
    all_candidates = set(vec_map.keys()) | set(graph_scores.keys())
    candidates: list[HybridResult] = []
    for nid in all_candidates:
        if nid not in graph.nodes:
            continue
        v_score = vec_map.get(nid, 0.0)
        g_score = graph_scores.get(nid, 0.0)
        candidates.append(HybridResult(
            node=graph.nodes[nid],
            vector_score=v_score,
            graph_score=g_score,
            combined_score=0.0,
            path=graph_paths.get(nid, ""),
        ))

    # 4. Нормализация score в [0, 1]
    max_v = max((c.vector_score for c in candidates), default=1.0) or 1.0
    max_g = max((c.graph_score for c in candidates), default=1.0) or 1.0
    for c in candidates:
        nv = c.vector_score / max_v
        ng = c.graph_score / max_g
        c.combined_score = round(vector_weight * nv + graph_weight * ng, 4)

    # 5. Сортировка и возврат
    candidates.sort(key=lambda c: -c.combined_score)
    return candidates[:top_k]


# --- Промпты ----------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты — ассистент по базам знаний и LLM-агентам. "
    "Отвечай ТОЛЬКО на основе предоставленного контекста. "
    "Если ответ есть в контексте — дай его кратко и укажи источники в формате [node_id]. "
    "Если ответа в контексте нет — честно скажи, что в памяти не нашлось информации. "
    "Не выдумывай. Отвечай на русском языке, кратко и по делу."
)

NO_CONTEXT_PROMPT = (
    "Ты — ассистент по базам знаний и LLM-агентам. "
    "По запросу в памяти ничего подходящего не найдено. "
    "Ответь на русском в 1-2 предложения: что в памяти не нашлось, "
    "и попроси переформулировать. Не выдумывай."
)


def build_messages(query: str, results: list[HybridResult], has_context: bool) -> list[dict]:
    """Собирает [system, user] с контекстом из гибридного поиска."""
    if has_context:
        parts = []
        for r in results:
            part = f"[{r.node.id}] {r.node.label} ({r.node.type})\n{r.node.text}"
            if r.path:
                part += f"\n  Графовый путь: {r.path}"
            part += f"\n  (вектор={r.vector_score:.3f}, граф={r.graph_score:.3f}, итого={r.combined_score:.3f})"
            parts.append(part)
        context_block = "\n\n".join(parts)
        user = f"Контекст:\n{context_block}\n\nВопрос:\n{query}"
        system = SYSTEM_PROMPT
    else:
        user = f"Вопрос:\n{query}"
        system = NO_CONTEXT_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- LLM вызов --------------------------------------------------------------

def answer_llm(client: openai.OpenAI, messages: list[dict]) -> str:
    """One-shot вызов LLM."""
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


# --- Порог релевантности ----------------------------------------------------

RELEVANCE_THRESHOLD = 0.3


# --- Оркестрация: один вопрос -----------------------------------------------

def run_query(
    query: str,
    client: openai.OpenAI,
    graph: GraphMemory,
    qdrant_mem,
    top_k: int,
    use_llm: bool = True,
    show_explain: bool = False,
) -> None:
    """Один прогон: гибридный поиск → ответ LLM."""
    results = hybrid_search(
        query, graph, qdrant_mem,
        top_k=top_k,
        vector_weight=VECTOR_WEIGHT,
        graph_weight=1.0 - VECTOR_WEIGHT,
        max_hop=GRAPH_MAX_HOP,
    )

    best = max((r.combined_score for r in results), default=0.0)
    has_context = best >= RELEVANCE_THRESHOLD
    ctx_results = results if has_context else []

    # Печать контекста
    if results:
        ctx_str = ", ".join(
            f"[{r.node.id}] {r.node.label} ({r.combined_score:.3f})"
            for r in results[:3]
        )
    else:
        ctx_str = "(ничего не найдено)"
    print(f"[поиск] {ctx_str}")

    if show_explain and results:
        for r in results[:3]:
            parts = [f"  [{r.node.id}] {r.node.label}"]
            parts.append(f"    вектор={r.vector_score:.3f}, граф={r.graph_score:.3f}, итого={r.combined_score:.3f}")
            if r.path:
                parts.append(f"    путь: {r.path}")
            print("\n".join(parts))

    if not has_context:
        print("[поиск] релевантных узлов не найдено (порог {0:.2f})".format(RELEVANCE_THRESHOLD))

    if not use_llm:
        return

    messages = build_messages(query, ctx_results, has_context)
    try:
        answer = answer_llm(client, messages)
        print(answer)
    except openai.APIError as e:
        print(f"[ошибка LLM] {e.__class__.__name__}: {str(e)[:300]}")
        return
    if has_context and ctx_results:
        sources = ", ".join(f"[{r.node.id}]" for r in ctx_results)
        print(f"[источники: {sources}]")


# --- CLI ---------------------------------------------------------------------

DEMO_QUESTIONS = [
    "Какое ДЗ использует RAG?",
    "Что связано с векторным поиском?",
    "Расскажи про дешёвую модель",
]


def run_demo(client, graph, qdrant_mem, top_k, use_llm: bool) -> None:
    print(f"Демо: {len(DEMO_QUESTIONS)} запросов")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n===== Демо {i}/{len(DEMO_QUESTIONS)} =====")
        print(f"Запрос: {q}")
        run_query(q, client, graph, qdrant_mem, top_k, use_llm)


def chat_loop(client, graph, qdrant_mem, top_k, use_llm: bool) -> None:
    print("Интерактивный режим. Выход: 'exit'/'quit'/'выход' или Ctrl-D.")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "выход"):
            break
        run_query(q, client, graph, qdrant_mem, top_k, use_llm)


def _make_client() -> openai.OpenAI:
    return openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# --- Самодиагностика --------------------------------------------------------

def selftest() -> int:
    """7 проверок без LLM и без Qdrant-сервера."""
    failures: list[str] = []
    kb_path = _resolve(KNOWLEDGE_FILE)

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e!r}")
            failures.append(name)

    # 1. knowledge.json валиден
    def t1_knowledge_valid() -> None:
        graph = load_knowledge(kb_path)
        assert len(graph.nodes) >= 5, f"меньше 5 узлов: {len(graph.nodes)}"
        all_out = sum(len(v) for v in graph.outgoing.values())
        assert all_out >= 3, f"меньше 3 рёбер: {all_out}"
        for n in graph.nodes.values():
            assert n.id and n.label and n.text, f"пустое поле у {n.id}"

    # 2. Граф построен, соседи корректны
    def t2_graph_neighbors() -> None:
        graph = load_knowledge(kb_path)
        neighbors = graph.neighbors("dz3", max_hop=1)
        assert "concept_rag" in neighbors, f"concept_rag не в соседях dz3: {neighbors}"
        assert "concept_streaming" in neighbors, f"concept_streaming не в соседях dz3: {neighbors}"
        assert len(neighbors) >= 2, f"меньше 2 соседей: {neighbors}"

    # 3. Моковый Qdrant: коллекция и поиск
    def t3_mock_qdrant() -> None:
        graph = load_knowledge(kb_path)
        embedder = _make_tfidf_embedder(graph)
        qmem = MockQdrantMemory(embedder.embed)
        qmem.upsert_nodes(list(graph.nodes.values()))
        assert len(qmem.points) == len(graph.nodes), "точки не загружены"
        results = qmem.search("RAG retrieval augmented", 5)
        assert results, "поиск вернул пусто"

    # 4. Векторный поиск: RAG → concept_rag
    def t4_vector_search_rag() -> None:
        graph = load_knowledge(kb_path)
        embedder = _make_tfidf_embedder(graph)
        qmem = MockQdrantMemory(embedder.embed)
        qmem.upsert_nodes(list(graph.nodes.values()))
        results = qmem.search("RAG retrieval augmented generation", 5)
        top_id = results[0][0]
        assert top_id == "concept_rag", f"топ не concept_rag: {top_id}"

    # 5. Графовый поиск: dz3 → ≥2 соседа
    def t5_graph_search() -> None:
        graph = load_knowledge(kb_path)
        neighbors = graph.neighbors("dz3", max_hop=1)
        assert len(neighbors) >= 2, f"dz3 имеет {len(neighbors)} соседей"

    # 6. Гибридный поиск: "какое ДЗ использует RAG" → dz3 в топ-3
    def t6_hybrid_search() -> None:
        graph = load_knowledge(kb_path)
        embedder = _make_tfidf_embedder(graph)
        qmem = MockQdrantMemory(embedder.embed)
        qmem.upsert_nodes(list(graph.nodes.values()))
        results = hybrid_search(
            "какое ДЗ использует RAG", graph, qmem,
            top_k=5, vector_weight=0.7, graph_weight=0.3, max_hop=2,
        )
        top_ids = [r.node.id for r in results[:3]]
        assert "dz3" in top_ids, f"dz3 не в топ-3 гибридного поиска: {top_ids}"

    # 7. Нормализация score
    def t7_score_normalization() -> None:
        graph = load_knowledge(kb_path)
        embedder = _make_tfidf_embedder(graph)
        qmem = MockQdrantMemory(embedder.embed)
        qmem.upsert_nodes(list(graph.nodes.values()))
        results = hybrid_search("эмбеддинг", graph, qmem, top_k=5)
        for r in results:
            assert 0.0 <= r.combined_score <= 1.0, \
                f"combined_score {r.combined_score} не в [0,1] для {r.node.id}"
        assert results, "результаты пустые"

    check("knowledge.json валиден (>=5 узлов, >=3 рёбра)", t1_knowledge_valid)
    check("граф: соседи dz3 включают concept_rag", t2_graph_neighbors)
    check("мок Qdrant: загрузка и поиск точек", t3_mock_qdrant)
    check("векторный поиск: RAG → concept_rag", t4_vector_search_rag)
    check("граф: dz3 >= 2 соседей", t5_graph_search)
    check("гибридный поиск: «ДЗ RAG» → dz3 в топ-3", t6_hybrid_search)
    check("нормализация combined_score в [0, 1]", t7_score_normalization)

    if failures:
        print(f"SELF-TEST: {len(failures)} упало: {', '.join(failures)}")
        return 1
    print("SELF-TEST: все проверки пройдены.")
    return 0


# --- Главная ----------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.ERROR),
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="DZ_4: память агента — векторный поиск (Qdrant) + графовые связи")
    parser.add_argument("question", nargs="*", help="одиночный вопрос")
    parser.add_argument("--demo", action="store_true", help="демо-запросы")
    parser.add_argument("--selftest", action="store_true", help="самодиагностика без LLM")
    parser.add_argument("--explain", action="store_true", help="показать детали поиска")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    # Загрузка графа
    graph = load_knowledge(_resolve(KNOWLEDGE_FILE))
    print(f"Загружено {len(graph.nodes)} узлов, "
          f"{sum(len(v) for v in graph.outgoing.values())} связей")

    # Qdrant память
    client = _make_client()
    try:
        embed_fn = lambda texts: embed_with_llm(client, texts)
        qdrant_mem = QdrantMemory(QDRANT_URL, QDRANT_COLLECTION, embed_fn)
        qdrant_mem.upsert_nodes(list(graph.nodes.values()))
        print(f"Qdrant: {QDRANT_COLLECTION} загружено")
    except Exception as e:
        print(f"[ошибка Qdrant] {e}. Использую мок.")
        qdrant_mem = MockQdrantMemory(lambda texts: [_random_embed(t) for t in texts])
        qdrant_mem.upsert_nodes(list(graph.nodes.values()))

    use_llm = True
    _print_header(graph, qdrant_mem)

    try:
        if args.demo:
            run_demo(client, graph, qdrant_mem, SEARCH_TOP_K, use_llm)
        elif args.question:
            q = " ".join(args.question)
            print(f"Запрос: {q}")
            run_query(q, client, graph, qdrant_mem, SEARCH_TOP_K,
                      use_llm, show_explain=args.explain)
        else:
            chat_loop(client, graph, qdrant_mem, SEARCH_TOP_K, use_llm)
    finally:
        if hasattr(qdrant_mem, "close"):
            qdrant_mem.close()


def _print_header(graph: GraphMemory, qdrant_mem) -> None:
    print(f"Агент DZ_4 | {len(graph.nodes)} узлов | {sum(len(v) for v in graph.outgoing.values())} связей "
          f"| модель={LLM_MODEL} | эмбеддинг={EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
