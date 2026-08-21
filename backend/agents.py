"""
TechNova Multi-Agent AI Analyst — core logic.
Barcha agentlar, supervisor, critic va LangGraph grafigi shu yerda.
"""

import os
import re
import time
import sqlite3
import subprocess
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, END

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY topilmadi — .env faylni tekshiring!")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=768,
)


# ---------------------------------------------------------------------------
# F1 — Shared state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    plan: str
    documents: List[str]
    sql_result: Optional[str]
    code_result: Optional[str]
    answer: Optional[str]
    steps: List[str]
    revisions: int
    last_ok: bool


# ---------------------------------------------------------------------------
# F2 — Ingestion & vector store
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "technova.db")

os.makedirs(DOCS_DIR, exist_ok=True)

_DEFAULT_DOCS = {
    "refund_policy.txt": """TechNova Qaytarish Siyosati
Mijozlar mahsulotni sotib olgandan keyin 30 kun ichida qaytarishi mumkin.
Qaytarish uchun mahsulot original qadoqda bo'lishi kerak.
Pul qaytarish 5-7 ish kuni ichida amalga oshiriladi.
Elektron mahsulotlar (noutbuklar, telefonlar) 14 kun ichida qaytariladi, agar zarar ko'rmagan bo'lsa.""",
    "product_faq.txt": """TechNova Mahsulotlar bo'yicha savol-javob
TechNova 2020-yilda tashkil topgan, dasturiy ta'minot va IT-uskunalar sotadi.
Asosiy mahsulotlar: bulutli xotira xizmati (CloudSafe), CRM tizimi (SalesPro), noutbuklar.
CloudSafe narxi: oyiga $9.99 dan boshlanadi, 1TB xotira bilan.
SalesPro CRM kichik va o'rta biznes uchun mo'ljallangan, oyiga $29 dan.
Texnik yordam 24/7 chat orqali, yoki support@technova.com email orqali.""",
    "churn_report.txt": """TechNova Mijozlar Yo'qotish (Churn) Tahlili — 2025 4-chorak
2025-yil 4-chorakda mijozlar yo'qotish darajasi 8.2% ni tashkil etdi, bu oldingi chorakdan yuqori.
Asosiy sabablar: (1) narxlar raqobatchilarga nisbatan yuqori deb topilgan, (2) mijozlar
texnik yordam javob berish tezligidan norozi bo'lgan, (3) SalesPro'da kerakli integratsiyalar yo'qligi.
Eng ko'p yo'qotish kichik biznes segmentida kuzatilgan — ular narxga sezgir.
Yirik korporativ mijozlar orasida yo'qotish past, atigi 2.1%.""",
}

for _fname, _content in _DEFAULT_DOCS.items():
    _fpath = os.path.join(DOCS_DIR, _fname)
    if not os.path.exists(_fpath):
        with open(_fpath, "w", encoding="utf-8") as f:
            f.write(_content)


def _build_vectorstore():
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader,
                              loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="technova_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    vs = QdrantVectorStore(client=client, collection_name="technova_docs", embedding=embeddings)
    vs.add_documents(chunks)
    return vs


vectorstore = _build_vectorstore()


def _build_memory_store():
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="conversation_memory",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    return QdrantVectorStore(client=client, collection_name="conversation_memory", embedding=embeddings)


memory_store = _build_memory_store()


def save_to_memory(question: str, answer: str):
    memory_store.add_texts([f"Savol: {question}\nJavob: {answer}"])


def recall_memory(question: str, k: int = 3):
    if not question:
        return []
    try:
        results = memory_store.similarity_search(question, k=k)
        return [r.page_content for r in results]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# F5 — SQLite (seeded)
# ---------------------------------------------------------------------------
def _build_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY, name TEXT, segment TEXT,
        monthly_revenue REAL, churned INTEGER)""")
    sample = [
        (1, "Alpha Corp", "Enterprise", 4200.0, 0),
        (2, "Beta LLC", "Small Business", 89.0, 1),
        (3, "Gamma Retail", "Small Business", 120.0, 1),
        (4, "Delta Tech", "Enterprise", 5100.0, 0),
        (5, "Epsilon Shop", "Small Business", 65.0, 1),
        (6, "Zeta Industries", "Mid-Market", 890.0, 0),
        (7, "Theta Solutions", "Small Business", 99.0, 0),
        (8, "Iota Group", "Enterprise", 6200.0, 0),
        (9, "Kappa Store", "Small Business", 75.0, 1),
        (10, "Lambda Inc", "Mid-Market", 950.0, 1),
    ]
    cur.execute("DELETE FROM customers")
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", sample)
    conn.commit()
    conn.close()


_build_database()


def _llm_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    return content.strip()


# ---------------------------------------------------------------------------
# F3 — Retriever agent
# ---------------------------------------------------------------------------
def retriever_agent(state: AgentState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {
        "documents": [d.page_content for d in docs],
        "steps": state["steps"] + ["retriever"],
    }


# ---------------------------------------------------------------------------
# F4 — Web agent (Tavily) — skips gracefully without a key
# ---------------------------------------------------------------------------
def web_agent(state: AgentState):
    if not TAVILY_API_KEY:
        return {"steps": state["steps"] + ["web(skipped)"]}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        hits = client.search(state["question"], search_depth="advanced")["results"]
        return {
            "documents": state["documents"] + [h["content"] for h in hits],
            "steps": state["steps"] + ["web"],
        }
    except Exception as e:
        # Never let a Tavily failure (bad key, rate limit, network) crash the graph
        # mid-stream — return cleanly so the supervisor can still finish the turn.
        return {"steps": state["steps"] + [f"web(error:{type(e).__name__})"]}


# ---------------------------------------------------------------------------
# F5 — Data / SQL agent (read-only guard)
# ---------------------------------------------------------------------------
DB_SCHEMA = """
Table: customers
Columns: id (INTEGER), name (TEXT), segment (TEXT: 'Enterprise'/'Mid-Market'/'Small Business'),
monthly_revenue (REAL), churned (INTEGER: 0=faol, 1=ketgan)
"""


def sql_agent(state: AgentState):
    prompt = f"""Sen SQL mutaxassisisan. Quyidagi jadval sxemasi asosida savolga javob beradigan
FAQAT bitta SQLite so'rovini yoz. Boshqa hech narsa yozma, faqat SQL kodini qaytar.

Sxema:
{DB_SCHEMA}

Savol: {state['question']}

SQL:"""
    sql_query = _llm_text(llm.invoke(prompt))
    sql_query = re.sub(r"```sql|```", "", sql_query).strip()

    if not sql_query.strip().lower().startswith("select"):
        return {
            "sql_result": f"Rad etildi: faqat SELECT so'rovlariga ruxsat berilgan. Model qaytargan: {sql_query}",
            "steps": state["steps"] + ["sql(rejected)"],
        }

    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(sql_query).fetchall()
        columns = [d[0] for d in conn.execute(sql_query).description]
        result_str = f"SQL: {sql_query}\nUstunlar: {columns}\nNatija: {result}"
    except Exception as e:
        result_str = f"SQL xatosi: {e}\nSo'rov: {sql_query}"
    finally:
        conn.close()

    return {"sql_result": result_str, "steps": state["steps"] + ["sql"]}


# ---------------------------------------------------------------------------
# F6 — Code agent (sandboxed: timeout + banned patterns)
# ---------------------------------------------------------------------------
_BANNED = ["import os", "import sys", "open(", "subprocess", "eval(", "exec(", "__import__"]


def code_agent(state: AgentState):
    prompt = f"""Sen Python dasturchisisan. Quyidagi savolga javob beruvchi Python kod yoz.
Natijani albatta print() bilan chiqar. Faqat kod yoz, ```python belgilarisiz.
Faqat standart kutubxonalardan foydalan, fayl/tarmoq operatsiyalari ishlatma.

Savol: {state['question']}

Kod:"""
    code = _llm_text(llm.invoke(prompt))
    code = re.sub(r"```python|```", "", code).strip()

    if any(b in code for b in _BANNED):
        return {
            "code_result": f"Rad etildi: xavfli kod aniqlandi.\nKod: {code}",
            "steps": state["steps"] + ["code(rejected)"],
        }

    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
        output = result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        output = "Xato: kod 5 soniyadan ko'p vaqt oldi (timeout)."

    return {"code_result": f"Kod:\n{code}\nNatija: {output}", "steps": state["steps"] + ["code"]}


# ---------------------------------------------------------------------------
# F7 — Supervisor / Router
# ---------------------------------------------------------------------------
class Route(BaseModel):
    next: str = Field(description="retriever, web, data, code yoki finish dan biri")


_QUANT_KEYWORDS = [
    "nechta", "necha ", "soni", "sonini", "eng ko'p", "eng kam",
    "eng yuqori", "eng past", "daromad", "reyting", "top",
]


def supervisor(state: AgentState):
    has_evidence = bool(state["documents"]) or bool(state["sql_result"]) or bool(state["code_result"])
    past_context = recall_memory(state["question"])
    memory_text = "\n".join(past_context) if past_context else "Yo'q"

    decision = llm.with_structured_output(Route).invoke(
        f"Savol: {state['question']}\n"
        f"O'tgan suhbatlardan tegishli kontekst: {memory_text}\n"
        f"Hozirgacha bajarilgan qadamlar: {state['steps']}\n"
        f"Yig'ilgan hujjatlar: {state['documents']}\n"
        f"SQL natijasi: {state['sql_result']}\n"
        f"Kod natijasi: {state['code_result']}\n"
        f"Ma'lumot allaqachon yig'ilganmi: {has_evidence}\n\n"
        f"Quyidagilardan birini tanla: retriever (hujjatlarda qidirish), "
        f"web (internetdan qidirish), data (SQL ma'lumotlar bazasi), "
        f"code (Python hisob-kitob), yoki finish.\n"
        f"MUHIM: agar savolga javob berish uchun yetarli ma'lumot allaqachon yig'ilgan bo'lsa, "
        f"albatta 'finish' tanla. Aks holda avval 'retriever' orqali hujjatlarda qidir.\n"
        f"MUHIM QOIDA: TechNova kompaniyasining o'z ma'lumotlari haqidagi savollar uchun HAR DOIM "
        f"avval 'retriever' tanla — 'web' faqat TechNova'ga aloqasi bo'lmagan tashqi ma'lumot uchun.\n"
        f"YANA BIR MUHIM QOIDA: agar savol hozirgi/so'nggi voqealar haqida bo'lsa (oxirgi, hozirgi, "
        f"so'nggi, natija, chempionat, saylov kabi so'zlar bilan), albatta 'web' ni tanla — hech qachon "
        f"o'zingning eski bilimingdan javob berma.\n"
        f"YANA BIR MUHIM QOIDA: agar savol aniq son, hisob, miqdor yoki reyting so'rasa (nechta, necha, "
        f"soni, eng ko'p, eng kam, eng yuqori, eng past, daromad kabi so'zlar bilan) va bu mijozlar "
        f"bazasiga tegishli bo'lsa, albatta 'data' (SQL) tanla — hujjatlardagi taxminiy ma'lumot yoki "
        f"o'zingning bilimingdan aniq raqam sifatida foydalanma."
    )

    updated_docs = state["documents"]
    if decision.next == "finish" and not has_evidence and past_context:
        updated_docs = state["documents"] + past_context

    # Deterministic safety net: the LLM router doesn't always follow the
    # quantitative-question rule above reliably. If the question clearly asks
    # for a count/ranking/amount tied to customer data and we haven't queried
    # SQL yet, force it — don't just hope the model picks it.
    q_lower = state["question"].lower()
    needs_quant_check = any(k in q_lower for k in _QUANT_KEYWORDS)
    if needs_quant_check and "sql" not in state["steps"] and "sql(rejected)" not in state["steps"]:
        forced_next = "data"
        return {
            "plan": forced_next,
            "documents": updated_docs,
            "steps": state["steps"] + [f"supervisor→{forced_next}(forced)"],
        }

    # Deterministic safety net #2: don't let the router retry the same agent
    # forever when it isn't producing evidence (e.g. a web search that returns
    # zero results). Count attempts by base agent name, ignoring any
    # "(error:...)" suffix, and force finish once an agent has been tried
    # twice without success — this is what actually makes the "bounded loop"
    # claim true, instead of relying only on the 15-step recursion limit.
    attempt_counts = {}
    for step in state["steps"]:
        base = step.split("(")[0]
        if base in ("retriever", "web", "data", "code"):
            attempt_counts[base] = attempt_counts.get(base, 0) + 1
    if decision.next in ("retriever", "web", "data", "code") and attempt_counts.get(decision.next, 0) >= 2:
        return {
            "plan": "finish",
            "documents": updated_docs,
            "steps": state["steps"] + [f"supervisor→finish(gave up on {decision.next})"],
        }

    return {
        "plan": decision.next,
        "documents": updated_docs,
        "steps": state["steps"] + [f"supervisor→{decision.next}"],
    }


# ---------------------------------------------------------------------------
# Generate — final answer
# ---------------------------------------------------------------------------
def generate(state: AgentState):
    has_any_evidence = bool(state["documents"]) or bool(state["sql_result"]) or bool(state["code_result"])
    if not has_any_evidence:
        answer = (
            "Kechirasiz, bu savol bo'yicha ishonchli ma'lumot topa olmadim "
            "(hujjatlar, ma'lumotlar bazasi yoki veb-qidiruv orqali). "
            "Boshqacharoq so'rab ko'rishingiz mumkin."
        )
        save_to_memory(state["question"], answer)
        return {"answer": answer}

    prompt = f"""Savol: {state['question']}

Yig'ilgan dalillar:
Hujjatlar: {state['documents']}
SQL natijasi: {state['sql_result']}
Kod natijasi: {state['code_result']}

Shu dalillar asosida, savolga aniq va qisqa javob yoz (o'zbek tilida)."""
    answer = _llm_text(llm.invoke(prompt))
    save_to_memory(state["question"], answer)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# F8 — Critic / Verifier
# ---------------------------------------------------------------------------
class Verdict(BaseModel):
    ok: bool = Field(description="Javob to'g'ri va dalillar bilan to'liq asoslanganmi")
    reason: str = Field(description="Qisqa sabab")


def critic(state: AgentState):
    verdict = llm.with_structured_output(Verdict).invoke(
        f"Savol: {state['question']}\n"
        f"Dalillar — Hujjatlar: {state['documents']}\n"
        f"SQL natijasi: {state['sql_result']}\n"
        f"Kod natijasi: {state['code_result']}\n"
        f"Berilgan javob: {state['answer']}\n\n"
        f"Bu javob to'g'rimi VA to'liq dalillar bilan asoslanganmi?"
    )
    return {
        "revisions": state["revisions"] + (0 if verdict.ok else 1),
        "last_ok": verdict.ok,
    }


# ---------------------------------------------------------------------------
# F9 — Graph wiring
# ---------------------------------------------------------------------------
def route_after_supervisor(state: AgentState):
    return state["plan"]


def route_after_critic(state: AgentState):
    if state.get("last_ok"):
        return "finish"
    if state["revisions"] >= 2:
        return "finish"
    return "revise"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_agent)
    g.add_node("web", web_agent)
    g.add_node("data", sql_agent)
    g.add_node("code", code_agent)
    g.add_node("generate", generate)
    g.add_node("critic", critic)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, {
        "retriever": "retriever", "web": "web", "data": "data",
        "code": "code", "finish": "generate",
    })
    g.add_edge("retriever", "supervisor")
    g.add_edge("web", "supervisor")
    g.add_edge("data", "supervisor")
    g.add_edge("code", "supervisor")
    g.add_edge("generate", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"finish": END, "revise": "supervisor"})
    return g.compile()


graph = build_graph()


def new_state(question: str) -> AgentState:
    return {
        "question": question, "plan": "", "documents": [],
        "sql_result": None, "code_result": None, "answer": None,
        "steps": [], "revisions": 0, "last_ok": False,
    }


def run_question(question: str) -> AgentState:
    return graph.invoke(new_state(question), config={"recursion_limit": 15})


def stream_question(question: str):
    """Generator: har bir agent qadamidan keyin (node_name, node_output) beradi."""
    state = new_state(question)
    for event in graph.stream(state, config={"recursion_limit": 15}):
        for node_name, node_output in event.items():
            yield node_name, node_output
