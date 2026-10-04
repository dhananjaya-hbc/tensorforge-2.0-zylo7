import { Suspense, lazy, useEffect, useState, type ComponentType } from "react"
import { KeyRound, Menu, Moon, ShieldCheck, Sun } from "lucide-react"
import { Toaster } from "@/components/ui/sonner"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ApiKeyDialog } from "@/components/api-key-dialog"
import { api, getKey, type Health } from "@/lib/api"
import { useTheme } from "@/lib/theme"
import { cn } from "@/lib/utils"

// Each page is its own chunk: the charts library only loads when Model Lab is opened.
const HomePage = lazy(() => import("@/pages/home"))
const TriagePage = lazy(() => import("@/pages/triage"))
const OpsBoardPage = lazy(() => import("@/pages/ops-board"))
const BatchLabPage = lazy(() => import("@/pages/batch-lab"))
const ModelLabPage = lazy(() => import("@/pages/model-lab"))

type Page = { id: string; label: string; Component: ComponentType }
const PAGES: Page[] = [
  { id: "home", label: "Overview", Component: HomePage },
  { id: "triage", label: "Live Triage", Component: TriagePage },
  { id: "ops", label: "Ops Board", Component: OpsBoardPage },
  { id: "batch", label: "Batch", Component: BatchLabPage },
  { id: "lab", label: "Model Lab", Component: ModelLabPage },
]

const fromHash = () => PAGES.find((p) => `#${p.id}` === window.location.hash)?.id ?? "home"

function useHealth() {
  const [health, setHealth] = useState<Health | null>(null)
  const [down, setDown] = useState(false)
  useEffect(() => {
    let alive = true
    const poll = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setDown(false)))
        .catch(() => alive && setDown(true))
    poll()
    const id = window.setInterval(poll, 15000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [])
  return { health, down }
}

export function Logo() {
  return (
    <a href="#home" className="flex items-center gap-2">
      <span className="grid size-8 place-items-center rounded-lg bg-gradient-to-br from-brand to-brand-2 text-white shadow-sm shadow-brand/30">
        <ShieldCheck className="size-4.5" />
      </span>
      <span className="text-[15px] font-semibold tracking-tight">Zylo7 Sentinel</span>
    </a>
  )
}

function StatusDot({ health, down }: { health: Health | null; down: boolean }) {
  const ok = health?.status === "ok" && !down
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="hidden items-center gap-2 rounded-full border bg-card px-2.5 py-1 text-xs text-muted-foreground lg:flex">
          <span className={cn("relative flex size-2", ok ? "text-ok" : "text-review")}>
            {ok && <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-50" />}
            <span className="relative inline-flex size-2 rounded-full bg-current" />
          </span>
          {down ? "Offline" : ok ? "Live" : "Loading"}
        </span>
      </TooltipTrigger>
      <TooltipContent className="font-mono text-xs">{health?.model_version ?? "model not loaded"}</TooltipContent>
    </Tooltip>
  )
}

export default function App() {
  const [page, setPage] = useState(fromHash)
  const [keyOpen, setKeyOpen] = useState(false)
  const [hasKey, setHasKey] = useState(() => Boolean(getKey()))
  const [menuOpen, setMenuOpen] = useState(false)
  const { health, down } = useHealth()
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    const onHash = () => {
      setPage(fromHash())
      window.scrollTo({ top: 0 })
    }
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])
  useEffect(() => {
    const open = () => setKeyOpen(true)
    window.addEventListener("sentinel:need-key", open)
    return () => window.removeEventListener("sentinel:need-key", open)
  }, [])

  const current = PAGES.find((p) => p.id === page) ?? PAGES[0]
  const dark = theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)

  const links = (vertical: boolean) =>
    PAGES.map(({ id, label }) => (
      <a
        key={id}
        href={`#${id}`}
        onClick={() => setMenuOpen(false)}
        aria-current={page === id ? "page" : undefined}
        className={cn(
          "rounded-full px-3.5 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground",
          vertical && "rounded-lg py-2",
          page === id && "bg-brand-soft font-medium text-brand",
        )}
      >
        {label}
      </a>
    ))

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-dvh">
        <header className="sticky top-0 z-30 border-b bg-background/75 backdrop-blur-xl">
          <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6">
            <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64 p-4">
                <SheetTitle className="sr-only">Navigation</SheetTitle>
                <Logo />
                <nav className="mt-6 flex flex-col gap-1">{links(true)}</nav>
              </SheetContent>
            </Sheet>
            <Logo />
            <nav className="ml-6 hidden items-center gap-1 md:flex">{links(false)}</nav>
            <div className="ml-auto flex items-center gap-2">
              <StatusDot health={health} down={down} />
              <Button variant="ghost" size="icon" aria-label="Toggle theme" onClick={() => setTheme(dark ? "light" : "dark")}>
                {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
              </Button>
              <Button variant={hasKey ? "outline" : "default"} size="sm" className="rounded-full" onClick={() => setKeyOpen(true)}>
                <KeyRound className="size-4" />
                <span className="hidden sm:inline">{hasKey ? "Key added" : "Add API key"}</span>
                {hasKey && <span className="size-1.5 rounded-full bg-ok" aria-label="connected" />}
              </Button>
            </div>
          </div>
        </header>
        <main className={cn(page === "home" ? "" : "mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:py-10")}>
          <Suspense fallback={<Skeleton className="mx-auto mt-10 h-[60vh] w-full max-w-7xl rounded-2xl" />}>
            <current.Component />
          </Suspense>
        </main>
      </div>
      <ApiKeyDialog open={keyOpen} onOpenChange={setKeyOpen} onChange={setHasKey} />
      <Toaster richColors closeButton position="bottom-right" />
    </TooltipProvider>
  )
}
