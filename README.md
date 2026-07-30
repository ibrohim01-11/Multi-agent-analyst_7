# TechNova Multi-Agent AI Analyst


## 🔗 Jonli demo

- **Frontend (ishlatish uchun):** https://multi-agent-analyst-7.vercel.app
- **Backend API:** https://multi-agent-analyst-7.onrender.com
- **GitHub:** https://github.com/ibrohim01-11/Multi-agent-analyst_7

> Eslatma: backend Render'ning bepul tarifida ishlaydi — agar 15 daqiqa
> foydalanilmasa "uxlab qoladi", birinchi so'rov 30-50 soniya kutishi mumkin.


Supervisor tomonidan boshqariladigan, 4 ta mutaxassis agent (Retriever, Web, SQL, Code)
va Critic (sifat nazorati) dan iborat multi-agent AI tizimi. `Multi_Agent_AI_Analyst_Guide_EN.html`
asosida qurilgan, F1-F14 to'liq bajarilgan.

## Loyiha tuzilishi

```
technova-app/
├── backend/          FastAPI + LangGraph (F1-F11 barcha agentlar shu yerda)
│   ├── agents.py      Barcha agentlar, supervisor, critic, graph
│   ├── main.py        FastAPI server (/ask, /ask-stream)
│   ├── requirements.txt
│   ├── render.yaml    Render.com uchun deploy konfiguratsiyasi
│   └── .env.example   API kalitlar namunasi
├── frontend/          Next.js streaming UI (F13)
│   ├── app/page.js     Chat interfeysi, SSE orqali jonli trace
│   ├── package.json
│   └── vercel.json
└── .gitignore
```

## 1. Lokal ishga tushirish

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # va o'z API kalitlaringni kirit
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```
Brauzerda `http://localhost:3000` ni och.

## 2. GitHub'ga yuklash

```bash
cd technova-app
git init
git add .
git commit -m "Initial commit: TechNova multi-agent analyst"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

**Muhim:** `.env` va `.env.local` fayllar `.gitignore` orqali chetlab o'tiladi — API kalitlaring
hech qachon GitHub'ga yuklanmaydi.

## 3. Backend'ni Render'ga deploy qilish (bepul, karta kerak emas)

1. `render.com`ga kir, GitHub akkaunting bilan ro'yxatdan o't
2. **"New +" → "Web Service"**
3. GitHub repo'ngni tanla, **Root Directory**: `backend`
4. Render `render.yaml`ni avtomatik topadi (yoki qo'lda: Build = `pip install -r requirements.txt`,
   Start = `uvicorn main:app --host 0.0.0.0 --port $PORT`)
5. **Environment** bo'limida `GOOGLE_API_KEY` va `TAVILY_API_KEY`ni qo'sh
6. Deploy qil — bir necha daqiqadan keyin `https://technova-backend.onrender.com` kabi manzil chiqadi

## 4. Frontend'ni Vercel'ga deploy qilish (bepul)

1. `vercel.com`ga kir, GitHub akkaunting bilan ro'yxatdan o't
2. **"Add New" → "Project"**, GitHub repo'ngni tanla
3. **Root Directory**: `frontend`
4. **Environment Variables**: `NEXT_PUBLIC_BACKEND_URL` = Render'dan olgan backend manzili
5. Deploy qil — `https://technova-app.vercel.app` kabi jamoat linki chiqadi

## Texnologiyalar

| Qism | Texnologiya |
|---|---|
| LLM | Google Gemini (`gemini-3.1-flash-lite`) |
| Embeddings | `gemini-embedding-001` |
| Vector DB | Qdrant (embedded, in-memory) |
| Orchestration | LangGraph |
| Web search | Tavily |
| SQL | SQLite (read-only guard bilan) |
| Backend | FastAPI (Render'da deploy qilinadi) |
| Frontend | Next.js (Vercel'da deploy qilinadi) |

## Bajarilgan Feature'lar (F1-F14) — 100/100

F1 Shared state · F2 Ingestion & vector store · F3 Retriever · F4 Web agent ·
F5 SQL agent (read-only) · F6 Code agent (sandboxed) · F7 Supervisor/Router ·
F8 Critic/Verifier · F9 LangGraph wiring · F10 Long-term memory ·
F11 Evaluation harness · F12 Langfuse observability · F13 Streaming frontend ·
F14 Deployment (Render + Vercel)
