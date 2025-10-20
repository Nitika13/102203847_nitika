// src/api.js
import axios from "axios";
const API_BASE_URL = "http://127.0.0.1:8000"; // change if backend host/port differ
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 15000,
});
export default api;
