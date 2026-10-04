import { ArrowRight, CircleCheck, Siren, UserRoundSearch } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { ApiError, type Segment } from "@/lib/api"
import { CATEGORY } from "@/lib/labels"
import { cn, pct } from "@/lib/utils"

/** Shows a friendly message for API errors; a 401 opens the API-key dialog. */
export function reportError(e: unknown) {
  if (e instanceof ApiError && e.status === 401) {
    window.dispatchEvent(new Event("sentinel:need-key"))
    toast.error("API key needed", { description: "Add the evaluation key to call the API." })
    return
  }
  if (e instanceof ApiError) {
    const detail = e.details.map((d) => [d.index !== undefined ? `#${d.index}` : "", d.field, d.issue].filter(Boolean).join(" ")).join("; ")
    toast.error(`${e.status} · ${e.code}`, { description: detail || e.message })
    return
  }
  toast.error("Request failed", { description: e instanceof Error ? e.message : String(e) })
}

export function CategoryMark({ category, size = "md" }: { category: string; size?: "sm" | "md" | "lg" }) {
  const info = CATEGORY[category]
  const Icon = info?.icon
  const box = { sm: "size-6 rounded-md", md: "size-9 rounded-lg", lg: "size-12 rounded-xl" }[size]
  const icon = { sm: "size-3.5", md: "size-4.5", lg: "size-6" }[size]
  return (
    <div className={cn("grid shrink-0 place-items-center border bg-brand-soft text-brand", box)}>
      {Icon && <Icon className={icon} />}
    </div>
  )
}

export function CategoryHeading({ category, team }: { category: string; team: string }) {
  return (
    <div className="flex items-center gap-3">
      <CategoryMark category={category} size="lg" />
      <div className="min-w-0">
        <div className="text-xl font-semibold tracking-tight">{CATEGORY[category]?.label ?? category}</div>
        <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <ArrowRight className="size-3.5" />
          <span className="truncate">{team}</span>
        </div>
      </div>
    </div>
  )
}

export function UrgencyBadge({ urgent, compact = false }: { urgent: boolean; compact?: boolean }) {
  return urgent ? (
    <Badge className="gap-1 border-transparent bg-urgent text-white hover:bg-urgent">
      <Siren className="size-3" /> Urgent
    </Badge>
  ) : compact ? null : (
    <Badge variant="outline" className="gap-1 text-muted-foreground">
      Not urgent
    </Badge>
  )
}

export function ReviewBadge({ review, compact = false }: { review?: boolean; compact?: boolean }) {
  if (review === undefined) return null
  return review ? (
    <Badge className="gap-1 border-transparent bg-review-soft text-review hover:bg-review-soft">
      <UserRoundSearch className="size-3" /> Needs review
    </Badge>
  ) : compact ? null : (
    <Badge className="gap-1 border-transparent bg-ok-soft text-ok hover:bg-ok-soft">
      <CircleCheck className="size-3" /> Auto-routed
    </Badge>
  )
}

export function confidenceTone(c: number) {
  return c >= 0.95 ? "bg-ok" : c >= 0.8 ? "bg-brand" : "bg-review"
}

export function ConfidenceMeter({ value }: { value: number }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-muted-foreground">Confidence</span>
        <span className="tabular text-2xl font-semibold tracking-tight">{pct(value, 1)}</span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-muted" role="meter" aria-valuenow={value} aria-valuemin={0} aria-valuemax={1}>
        <div className={cn("h-full rounded-full transition-all duration-500", confidenceTone(value))} style={{ width: pct(value, 1) }} />
        <div className="absolute inset-y-0 w-px bg-foreground/40" style={{ left: "80%" }} title="review threshold 0.80" />
      </div>
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>0</span>
        <span style={{ marginLeft: "62%" }}>review below 80%</span>
        <span>100</span>
      </div>
    </div>
  )
}

const LEGEND: { kind: Segment["kind"]; label: string }[] = [
  { kind: "signal", label: "Issue the model reads" },
  { kind: "filler", label: "Boilerplate (ignored)" },
  { kind: "agent", label: "Agent turn (ignored)" },
  { kind: "injection", label: "Injection (neutralised)" },
  { kind: "quoted", label: "Quoted history (ignored)" },
]

export function Evidence({ segments }: { segments: Segment[] }) {
  const present = new Set(segments.map((s) => s.kind))
  return (
    <div className="space-y-3">
      <p className="whitespace-pre-wrap break-words text-[15px] leading-8">
        {segments.map((s, i) =>
          s.kind === "space" ? <span key={i}>{s.text}</span> : <span key={i} className={`seg-${s.kind}`}>{s.text}</span>,
        )}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 border-t pt-3 text-xs text-muted-foreground">
        {LEGEND.filter((l) => present.has(l.kind)).map((l) => (
          <span key={l.kind} className="flex items-center gap-1.5">
            <span className={`seg-${l.kind} inline-block h-3 w-5 !p-0`} aria-hidden />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  )
}
