// src/pages/Recommend.jsx
import React, { useState, useRef } from "react";
import api from "src\api.js";
import { FiSearch, FiClock, FiChevronRight } from "react-icons/fi";
import { motion } from "framer-motion";

const truncate = (s, n = 140) => {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
};

export default function Recommend() {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [lastQuery, setLastQuery] = useState(null);
  const inputRef = useRef();

  async function search(k = 12) {
    const q = (prompt || "").trim();
    if (!q) {
      setMessage('Type a short query (example: "affordable wooden chair" or just "shoes").');
      inputRef.current?.focus();
      return;
    }

    setMessage("");
    setLoading(true);
    setResults([]);
    setLastQuery(q);
    try {
      // IMPORTANT: backend routes are mounted under /api — use /api/recommend/
      const res = await api.post("/api/recommend/", { prompt: q, k });
      const arr = Array.isArray(res?.data?.results) ? res.data.results : [];
      setResults(arr);
      if (!arr.length) setMessage("No matches found. Try a different phrase.");
    } catch (err) {
      console.error("Search error:", err);
      // More helpful error message for dev
      if (err.response) {
        // backend returned an error response
        console.error("Response data:", err.response.data);
        setMessage(`Server error: ${err.response.status} ${err.response.data?.detail ?? ""}`);
      } else if (err.request) {
        // request made but no response
        setMessage("No response from server — is backend running and reachable?");
      } else {
        setMessage("Error: " + (err.message || "Unknown error"));
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 18 }}>
      <aside className="card" style={{ height: "fit-content", padding: 18 }}>
        <h3 style={{ margin: 0 }}>Search products</h3>
        <p className="small" style={{ marginTop: 8 }}>
          Enter a short prompt (product type, attributes, or use-case). The recommender will return similar items.
        </p>

        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <input
            ref={inputRef}
            className="input"
            value={prompt}
            placeholder='e.g. "budget running shoes"'
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") search();
            }}
          />
          <button className="btn btn-primary" onClick={() => search()}>
            <FiSearch style={{ marginRight: 8 }} /> {loading ? "Searching..." : "Search"}
          </button>
        </div>

        <div style={{ marginTop: 14 }}>
          <div className="small" style={{ marginBottom: 8 }}>Quick tips</div>
          <ul className="small" style={{ paddingLeft: 18, marginTop: 0 }}>
            <li>Be specific: "leather office chair" &gt; "chair"</li>
            <li>Try adjectives: "affordable / premium / kids / outdoor"</li>
            <li>Use category or color: "blue lamp", "wooden table"</li>
          </ul>
        </div>

        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <button className="btn" onClick={() => { setPrompt("shoes"); setTimeout(() => search(8), 50); }}>Shoes</button>
          <button className="btn" onClick={() => { setPrompt("wooden table"); setTimeout(() => search(8), 50); }}>Wooden table</button>
        </div>

        <div style={{ marginTop: 16 }} className="small">
          {loading && <div style={{ display: "flex", alignItems: "center", gap: 8 }}><FiClock /> Searching...</div>}
          {!loading && lastQuery && <div>Last search: <strong>{lastQuery}</strong> — {results.length} result{results.length !== 1 ? "s" : ""}</div>}
          {!loading && !lastQuery && <div>Ready to search</div>}
        </div>
      </aside>

      <main>
        <h1 style={{ marginTop: 0 }}>Recommender</h1>
        <p className="small" style={{ marginTop: 4 }}>
          Results are ranked by semantic similarity.
        </p>

        {message && <div className="card" style={{ padding: 12, marginTop: 12 }}><strong>{message}</strong></div>}

        <div className="results-grid" style={{ marginTop: 12 }}>
          {results.map((r) => (
            <motion.div
              key={r.id}
              className="card"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ y: -6 }}
            >
              <div style={{ display: "flex", gap: 12 }}>
                <div style={{
                  width: 120, height: 90, borderRadius: 8,
                  background: "linear-gradient(135deg, rgba(190,231,212,0.9), rgba(255,220,168,0.9))",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#fff", fontWeight: 700, flexShrink: 0
                }}>
                  {r.metadata?.title ? (r.metadata.title[0] || "P") : "P"}
                </div>

                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <div>
                      <div className="card-title">{r.metadata?.title ?? r.id}</div>
                      <div className="small" style={{ color: "var(--muted)", marginTop: 6 }}>
                        {r.metadata?.brand ? `${r.metadata.brand} — ` : ""}{r.metadata?.store ?? ""}
                      </div>
                    </div>

                    <div style={{ textAlign: "right", minWidth: 96 }}>
                      <div style={{ fontWeight: 700 }}>{r.price_display ?? "N/A"}</div>
                      <div className="small" style={{ marginTop: 10 }}>Score: {(Number(r.score) || 0).toFixed(3)}</div>
                    </div>
                  </div>

                  <div className="card-desc" style={{ marginTop: 10 }}>
                    {truncate(r.generated_description ?? r.metadata?.description ?? "—", 200)}
                  </div>

                  <div style={{ marginTop: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div className="small" style={{ color: "var(--muted)" }}>ID: {r.id}</div>
                    <div>
                      <a className="btn" style={{ padding: "6px 10px", borderRadius: 10 }} href="#" onClick={(e) => { e.preventDefault(); alert("Open product details: " + (r.metadata?.title ?? r.id)); }}>
                        View <FiChevronRight style={{ marginLeft: 6 }} />
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
}
