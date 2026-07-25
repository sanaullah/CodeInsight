import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

const root = document.getElementById("app-root");
if (!root) throw new Error("CodeInsight frontend root is missing");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
