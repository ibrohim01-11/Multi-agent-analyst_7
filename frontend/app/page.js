"use client";

import { useState, useRef } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const NODE_LABELS = {
  supervisor: "🧭 Supervisor",
  retriever: "📄 Retriever",
  web: "🌐 Web",
  data: "🗄️ SQL",
  code: "🧮 Code",
  generate: "✍️ Generate",
  critic: "✅ Critic",
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const esRef = useRef(null);

  function ask() {
    if (!question.trim() || loading) return;
    setSteps([]);
    setAnswer("");
    setLoading(true);

    const url = `${BACKEND_URL}/ask-stream?question=${encodeURIComponent(question)}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSteps((prev) => [...prev, data]);
      if (data.answer) setAnswer(data.answer);
    };

    es.addEventListener("done", () => {
      es.close();
      setLoading(false);
    });

    es.onerror = () => {
      es.close();
      setLoading(false);
    };
  }

  return (
    <main style={{ maxWidth: 760, margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ fontSize: 28, marginBottom: 4 }}>🤖 TechNova Multi-Agent AI Analyst</h1>
      <p style={{ color: "#9aa4bf", marginBottom: 28 }}>
        Savolingizni yozing — agentlar (Supervisor, Retriever, SQL, Code, Critic) jonli ishlaydi.
      </p>

      <div style={{ display: "flex", gap: 10 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Masalan: Qaysi segmentda eng ko'p mijoz ketib qolgan?"
          style={{
            flex: 1,
            padding: "12px 14px",
            borderRadius: 10,
            border: "1px solid #2a2f45",
            background: "#161a2c",
            color: "#eaeef7",
            fontSize: 15,
          }}
        />
        <button
          onClick={ask}
          disabled={loading}
          style={{
            padding: "12px 22px",
            borderRadius: 10,
            border: "none",
            background: loading ? "#5b4a2a" : "#e07a2f",
            color: "white",
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Ishlayapti..." : "Yuborish"}
        </button>
      </div>

      <div style={{ marginTop: 30, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div style={{ background: "#161a2c", borderRadius: 12, padding: 18, minHeight: 160 }}>
          <h3 style={{ marginTop: 0, color: "#9aa4bf", fontSize: 13, textTransform: "uppercase" }}>
            Agentlar jarayoni (jonli)
          </h3>
          {steps.length === 0 && <p style={{ color: "#555b73" }}>Hali savol yuborilmadi.</p>}
          {steps.map((s, i) => (
            <div key={i} style={{ marginBottom: 6, fontSize: 14 }}>
              {NODE_LABELS[s.node] || s.node} ishladi
              {s.plan && <span style={{ color: "#9aa4bf" }}> → yo'nalish: <code>{s.plan}</code></span>}
            </div>
          ))}
        </div>

        <div style={{ background: "#161a2c", borderRadius: 12, padding: 18, minHeight: 160 }}>
          <h3 style={{ marginTop: 0, color: "#9aa4bf", fontSize: 13, textTransform: "uppercase" }}>
            Yakuniy javob
          </h3>
          <p style={{ fontSize: 15, lineHeight: 1.5 }}>{answer || "—"}</p>
        </div>
      </div>
    </main>
  );
}
