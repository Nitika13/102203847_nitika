import React, { useState } from "react";
import axios from "axios";

export default function Recommend() {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/api/recommend/", {
        prompt,
        k: 5
      });
      setResults(res.data.results || []);
    } catch (err) {
      console.error("Search error:", err);
      alert("Backend error — check console");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Recommender</h1>
      <input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      <button onClick={handleSearch}>{loading ? "Searching..." : "Search"}</button>

      <div style={{ marginTop: 20 }}>
        {results.map((r) => (
          <div key={r.id}>
            <strong>{r.metadata?.title ?? r.id}</strong>
            <div>{r.generated_description}</div>
            <div>Score: {r.score}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
