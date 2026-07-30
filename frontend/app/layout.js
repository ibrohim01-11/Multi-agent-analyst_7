export const metadata = {
  title: "TechNova AI Analyst",
  description: "Multi-agent AI analyst — supervisor, retriever, SQL, code, critic",
};

export default function RootLayout({ children }) {
  return (
    <html lang="uz">
      <body style={{ margin: 0, fontFamily: "system-ui, -apple-system, sans-serif", background: "#0f1220", color: "#eaeef7" }}>
        {children}
      </body>
    </html>
  );
}
