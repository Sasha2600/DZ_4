# DZ_4: Память агента — векторный поиск + граф связей

Агент, который повышает точность ответа за счёт **гибридного поиска** по памяти:

1. **Векторный поиск** — узлы знаний хранятся в Qdrant (Docker, `localhost:6333`),
   эмбеддинги — через OpenAI-совместимый сервер (LM Studio), сходство — косинусное.
2. **Граф связей** — отношения между узлами (`knowledge.json` → adjacency list),
   расширение кандидатов по соседству (BFS до 2 hop).
3. **Гибридный скоринг** — `combined = 0.7 * norm_vector + 0.3 * norm_graph`,
   нормализация обоих слагаемых в [0, 1].
4. **Ответ LLM** — one-shot, контекст-блок из найденных узлов + графовые пути,
   источники `[node_id]`.

Пайплайн на вопрос: `Qdrant-поиск (top-k) → расширение по графу → нормализация score →
system+context-промпт → ответ LLM → источники [node_id]`.

## Структура

```
agent.py              # агент: граф, Qdrant-память, гибридный поиск, LLM, CLI, selftest
memory/knowledge.json # память: 10 узлов (homework/topic/model/concept) + 10 связей
requirements.txt      # openai, python-dotenv, qdrant-client (pinned)
.env.example          # шаблон конфигурации
plan.md               # план реализации
```

## Требования

- Python 3.12+ (разработано и проверено на 3.13).
- OpenAI-совместимый LLM-сервер (LM Studio, дефолт `http://localhost:1234/v1`)
  с чат-моделью и embedding-моделью.
- Qdrant — Docker-контейнер на `localhost:6333`:

  ```bash
  docker compose up -d            # или: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
  ```

  Если Qdrant или LLM недоступны — агент не падает: поиск деградирует до мок-памяти
  (детерминированные псевдо-векторы), на LLM — дружелюбное `[ошибка LLM]`.

## Установка

```bash
cd DZ_4
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # затем вписать LLM_MODEL / EMBEDDING_MODEL из LM Studio
```

## Запуск

```bash
# интерактивный режим (exit/quit/выход — выход)
.venv/bin/python agent.py

# один вопрос
.venv/bin/python agent.py "какое ДЗ использует RAG?"

# сценарный прогон: 3 демо-запроса
.venv/bin/python agent.py --demo

# с деталями поиска (векторный/графовый score, путь в графе)
.venv/bin/python agent.py --explain "Что связано с векторным поиском?"

# самодиагностика без LLM и без Qdrant (7 проверок)
.venv/bin/python agent.py --selftest
```

## Сценарий для видео/скрина

1. `docker start qdrant` (или поднять контейнер) — показать `http://localhost:6333/collections`.
2. `.venv/bin/python agent.py --demo --explain`:
   - **Демо 1** — «Какое ДЗ использует RAG?»: вектор → `concept_rag`,
     граф по связи «реализует» поднимает `dz3` в топ, ответ со ссылками `[dz3]`.
   - **Демо 2** — «Что связано с векторным поиском?»: вектор → `topic_vector`,
     граф (2 hop) → `topic_embedding`, `concept_rag`, `topic_graph`.
   - **Демо 3** — «Расскажи про дешёвую модель»: топ-векторный хит `model_cheap`
     (сам в контексте), граф-расширение поднимает `dz1` в топ,
     ответ описывает модель со ссылкой `[model_cheap]`.
3. Опционально: `--selftest` — 7 зелёных проверок без внешних сервисов.
4. Опционально: остановить Qdrant (`docker stop qdrant`) и повторить вопрос —
   агент сам переключится на мок-память (строка `[ошибка Qdrant] ... Использую мок.`),
   а не упадёт.

## Архитектура

```
CLI (agent.py): интерактив | "вопрос" | --demo | --selftest | --explain
  |
  v
run_query(query)
  |  1. QdrantMemory.search(query, top_k)
  |     |-- Qdrant (Docker, cosine) + embed_with_llm() через LM Studio
  |     +-- MockQdrantMemory (фолбэк без Qdrant/LLM: детерминированные векторы)
  |
  |  2. GraphMemory.neighbors(hit, max_hop=2)   [BFS, bidirectional]
  |     граф. score = 1/distance; путь — BFS get_path()
  |
  |  3. hybrid_search(): объединяет кандидатов,
  |     нормализует vector_score и graph_score в [0, 1],
  |     combined = 0.7*vec + 0.3*graph, сортировка, top_k
  |
  |  4. build_messages(): system + user
  |     с контекстом:  "[node_id] label (type)\n text\n Графовый путь: ... (scores)"
  |     без контекста: NO_CONTEXT_PROMPT (вежливый отказ, не выдумывать)
  |
  v
[источники: [node_id] ...]
```

Ключевые решения:

- **Единый источник данных.** `knowledge.json` — и для графа (в памяти),
  и для Qdrant (узлы upsert-ятся с payload `node_id/label/type/text`).
- **Два сигнала релевантности.** Векторный — семантика; графовый — структура.
  Вопрос «какое ДЗ использует RAG» векторно бьёт в `concept_rag`, но правильный
  ответ `dz3` даёт именно связь `dz3 → concept_rag «реализует»`.
- **Нормализация.** Векторный скор (косинус, [-1, 1]) сначала обрезают до
  [0, 1] (отрицательное сходство — «не подходит», иначе деление на максимум
  вывело бы `combined_score` за диапазон); затем оба слагаемых (вектор и
  граф, 1/distance) масштабируются на максимум по кандидатам —
  `combined_score` гарантированно в [0, 1].
- **Устойчивость.** Qdrant недоступен → мок-память (демо-поиск работает);
  LLM недоступен → `[ошибка LLM]`, а не traceback; selftest не требует ни LLM,
  ни Qdrant (мок Qdrant + TF-IDF эмбеддер).

## Конфигурация (`.env`)

| Переменная | Дефолт | Назначение |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | адрес OpenAI-совместимого сервера |
| `LLM_API_KEY` | `lm-studio` | ключ (для LM Studio — любое непустое) |
| `LLM_MODEL` | `google/gemma-4-12b-qat` | Model Identifier чат-модели |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `120` | таймаут запроса к LLM (one-shot на локальной модели долговат) |
| `EMBEDDING_MODEL` | `text-embedding-qwen3-embedding-0.6b` | Model Identifier embedding-модели |
| `QDRANT_URL` | `http://localhost:6333` | адрес Qdrant |
| `QDRANT_COLLECTION` | `dz4_memory` | имя коллекции |
| `EMBEDDING_DIM` | `1024` | размерность вектора (должна совпадать с моделью) |
| `KNOWLEDGE_FILE` | `memory/knowledge.json` | путь к данным |
| `SEARCH_TOP_K` | `5` | сколько результатов возвращать |
| `VECTOR_WEIGHT` | `0.7` | вес векторного слагаемого (остальное — граф) |
| `GRAPH_MAX_HOP` | `2` | максимальная глубина обхода графа |
| `LOG_LEVEL` | `ERROR` | уровень логирования |

## Selftest (без LLM)

`.venv/bin/python agent.py --selftest` — 7 проверок, LLM-сервер и Qdrant **не нужны**:

1. `knowledge.json` валиден (≥5 узлов, ≥3 рёбра, поля не пустые);
2. граф построен: соседи `dz3` включают `concept_rag` и `concept_streaming`;
3. мок Qdrant: загрузка точек и поиск не пустой;
4. векторный поиск: запрос «RAG retrieval augmented generation» → `concept_rag` на 1 месте;
5. графовый поиск: у `dz3` ≥2 соседей на 1 hop;
6. гибридный поиск: «какое ДЗ использует RAG» → `dz3` в топ-3;
7. `combined_score` всех результатов в [0, 1].

Успех = `SELF-TEST: все проверки пройдены.` и код возврата 0.