import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "@fontsource-variable/inter"
import "@fontsource-variable/jetbrains-mono"
import "@fontsource/noto-sans-sinhala/400.css"
import "@fontsource/noto-sans-sinhala/600.css"
import "@fontsource/noto-sans-tamil/400.css"
import "@fontsource/noto-sans-tamil/600.css"
import "./index.css"
import App from "./App"
import { ThemeProvider } from "@/lib/theme"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
)
