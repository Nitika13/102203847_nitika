import { Link } from "react-router-dom";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center">
      <h1>AI Product Recommender - made by Nitika</h1>
      <Link to="/recommend">Go to Recommender</Link>
    </div>
  );
}
