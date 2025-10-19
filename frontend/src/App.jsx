import { Link } from "react-router-dom";

export default function App() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 text-center px-4">
      <h1 className="text-4xl md:text-5xl font-extrabold text-gray-800 mb-4">
        🛍️ AI Product Recommender
      </h1>

      <p className="text-gray-600 text-lg md:text-xl max-w-2xl mb-8">
        Discover products tailored to your needs using AI-powered search and smart recommendations.
      </p>

      <div className="flex flex-col sm:flex-row gap-4">
        <Link
          to="/recommend"
          className="bg-blue-600 text-white px-8 py-3 rounded-2xl shadow-md hover:bg-blue-700 transition duration-200"
        >
          Go to Recommender
        </Link>

        <Link
          to="/analytics"
          className="bg-white text-blue-700 border border-blue-600 px-8 py-3 rounded-2xl shadow-md hover:bg-blue-50 transition duration-200"
        >
          View Analytics
        </Link>
      </div>

      <footer className="mt-12 text-gray-500 text-sm">
        Built with ❤️ using FastAPI + React + Vite
      </footer>
    </div>
  );
}
