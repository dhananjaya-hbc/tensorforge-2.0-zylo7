import { Suspense, lazy, useEffect, useState, type ComponentType } from "react"
import { FlaskConical, KeyRound, LayoutGrid, Layers, Menu, Monitor, Moon, ScanText, ShieldCheck, Sun } from "lucide-react"
import { Toaster } from "@/components/ui/sonner"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ApiKeyDialog } from "@/components/api-key-dialog"
import { api, getKey, type Health } from "@/lib/api"
import { useTheme } from "@/lib/theme"
import { cn } from "@/lib/utils"
import { Skeleton } from "@/components/ui/skeleton"

// Each page is its own chunk: the charts library only loads when Model Lab is opened.
const TriagePage = lazy(() => import("@/pages/triage"))
const OpsBoardPage = lazy(() => import("@/pages/ops-board"))
const BatchLabPage = lazy(() => import("@/pages/batch-lab"))
const ModelLabPage = lazy(() => import("@/pages/model-lab"))

type Page = { id: string; label: string; blurb: string; icon: ComponentType<{ className?: string }>; Component: ComponentType }

const PAGES: Page[] = [
  { id: "triage", label: "Live Triage", blurb: "Classify one ticket and see what the model read", icon: ScanText, Component: TriagePage },
  { id: "ops", label: "Ops Board", blurb: "Tickets routed to support teams in real time", icon: LayoutGrid, Component: OpsBoardPage },
  { id: "batch", label: "Batch Lab", blurb: "Run a file of tickets as an async job", icon: Layers, Component: BatchLabPage },
  { id: "lab", label: "Model Lab", blurb: "How the model was trained, validated and improved", icon: FlaskConical, Component: ModelLabPage },
]

const fromHash = () => PAGES.find((p) => `#${p.id}` === window.location.hash)?.id ?? "triage"

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

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid size-8 place-items-center rounded-lg bg-brand text-brand-foreground shadow-sm">
        <ShieldCheck className="size-4.5" />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight">Triage Sentinel</div>
        <div className="text-[11px] text-muted-foreground">RideEat Command Center</div>
      </div>
    </div>
  )
}

function ThemeSwitch() {
  const { theme, setTheme } = useTheme()
  const options = [
    { id: "light", icon: Sun, label: "Light" },
    { id: "dark", icon: Moon, label: "Dark" },
    { id: "system", icon: Monitor, label: "System" },
  ] as const
  return (
    <div className="flex rounded-lg border bg-background p-0.5" role="radiogroup" aria-label="Theme">
      {options.map(({ id, icon: Icon, label }) => (
        <Tooltip key={id}>
          <TooltipTrigger asChild>
            <button
              role="radio"
              aria-checked={theme === id}
              aria-label={label}
              onClick={() => setTheme(id)}
              className={cn(
                "grid size-7 place-items-center rounded-md text-muted-foreground transition-colors hover:text-foreground",
                theme === id && "bg-muted text-foreground",
              )}
            >
              <Icon className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent>{label}</TooltipContent>
        </Tooltip>
      ))}
    </div>
  )
}

function StatusLine({ health, down }: { health: Health | null; down: boolean }) {
  const ok = health?.status === "ok" && !down
  return (
    <div className="space-y-1 rounded-lg border bg-background/60 p-3 text-xs">
      <div className="flex items-center gap-2 font-medium">
        <span className={cn("relative flex size-2", ok ? "text-ok" : "text-review")}>
          {ok && <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-50" />}
          <span className="relative inline-flex size-2 rounded-full bg-current" />
        </span>
        {down ? "API unreachable" : ok ? "API online" : "Model loading"}
      </div>
      <div className="truncate font-mono text-[11px] text-muted-foreground" title={health?.model_version ?? ""}>
        {health?.model_version ?? "—"}
      </div>
    </div>
  )
}

function Nav({ page, onNavigate }: { page: string; onNavigate: (id: string) => void }) {
  return (
    <nav className="space-y-1">
      {PAGES.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onNavigate(id)}
          aria-current={page === id ? "page" : undefined}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground",
            page === id && "bg-sidebar-accent font-medium text-foreground",
          )}
        >
          <Icon className={cn("size-4", page === id && "text-brand")} />
          {label}
        </button>
      ))}
    </nav>
  )
}

export default function App() {
  const [page, setPage] = useState(fromHash)
  const [keyOpen, setKeyOpen] = useState(false)
  const [hasKey, setHasKey] = useState(() => Boolean(getKey()))
  const [menuOpen, setMenuOpen] = useState(false)
  const { health, down } = useHealth()

  useEffect(() => {
    const onHash = () => setPage(fromHash())
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])
  useEffect(() => {
    const open = () => setKeyOpen(true)
    window.addEventListener("sentinel:need-key", open)
    return () => window.removeEventListener("sentinel:need-key", open)
  }, [])

  const navigate = (id: string) => {
    window.location.hash = id
    setPage(id)
    setMenuOpen(false)
  }
  const current = PAGES.find((p) => p.id === page) ?? PAGES[0]

  const sidebar = (
    <div className="flex h-full flex-col gap-6 p-4">
      <Brand />
      <Nav page={page} onNavigate={navigate} />
      <div className="mt-auto space-y-3">
        <StatusLine health={health} down={down} />
        <ThemeSwitch />
      </div>
    </div>
  )

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-dvh md:grid md:grid-cols-[15rem_1fr]">
        <aside className="sticky top-0 hidden h-dvh border-r bg-sidebar md:block">{sidebar}</aside>
        <div className="min-w-0">
          <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-background/80 px-4 backdrop-blur md:px-8">
            <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64 bg-sidebar p-0">
                <SheetTitle className="sr-only">Navigation</SheetTitle>
                {sidebar}
              </SheetContent>
            </Sheet>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold">{current.label}</h1>
              <p className="hidden truncate text-xs text-muted-foreground sm:block">{current.blurb}</p>
            </div>
            <div className="ml-auto">
              <Button variant={hasKey ? "outline" : "default"} size="sm" onClick={() => setKeyOpen(true)}>
                <KeyRound className="size-4" />
                {hasKey ? (
                  <span className="flex items-center gap-1.5">
                    API key <span className="size-1.5 rounded-full bg-ok" aria-label="connected" />
                  </span>
                ) : (
                  "Add API key"
                )}
              </Button>
            </div>
          </header>
          <main className="mx-auto w-full max-w-[1400px] px-4 py-6 md:px-8 md:py-8">
            <Suspense fallback={<Skeleton className="h-[60vh] w-full rounded-xl" />}>
              <current.Component />
            </Suspense>
          </main>
        </div>
      </div>
      <ApiKeyDialog open={keyOpen} onOpenChange={setKeyOpen} onChange={setHasKey} />
      <Toaster richColors closeButton position="bottom-right" />
    </TooltipProvider>
  )
}
