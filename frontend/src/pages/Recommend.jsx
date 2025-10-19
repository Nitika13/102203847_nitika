import { useState } from "react";
import axios from "axios";

export default function Recommend() {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await axios.post("http://127.0.0.1:8000/recommend/", {
        prompt,
        k: 5,
      });
      setResults(res.data.results || []);
    } catch (err) {
      console.error(err);
      setError("Server not responding. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
        🛍️ AI Product Recommender
      </h1>

      <div className="flex justify-center mb-6">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Search for products..."
          className="w-1/2 p-3 border border-gray-300 rounded-l-lg focus:outline-none"
        />
        <button
          onClick={handleSearch}
          className="bg-blue-600 text-white px-6 rounded-r-lg hover:bg-blue-700"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="text-red-500 text-center">{error}</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-6">
        {results.map((item) => (
          <div
            key={item.id}
            className="bg-white shadow-md p-4 rounded-2xl hover:shadow-lg transition"
          >
            <h2 className="text-lg font-semibold text-gray-800">
              {item.metadata?.title || "Untitled"}
            </h2>
            <p className="text-gray-500 text-sm">
              Brand: {item.metadata?.brand || "Unknown"}
            </p>
            <p className="text-gray-700 mt-2 text-sm">
              {item.generated_description}
            </p>
            <p className="mt-3 text-blue-700 font-semibold">
              Score: {item.score?.toFixed(3)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
