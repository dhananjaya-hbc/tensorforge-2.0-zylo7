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
    <div className={cn("grid shrink-0 place-items-center ring-1 ring-inset", info?.chip, box)}>
      {Icon && <Icon className={icon} />}
    </div>
  )
}

export function CategoryChip({ category, className }: { category: string; className?: string }) {
  const info = CATEGORY[category]
  const Icon = info?.icon
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset", info?.chip, className)}>
      {Icon && <Icon className="size-3.5" />}
      {info?.label ?? category}
    </span>
  )
}

/** Radial gauge; the small tick marks the 0.80 human-review threshold. */
export function ConfidenceRing({ value, size = 112 }: { value: number; size?: number }) {
  const r = 42
  const c = 2 * Math.PI * r
  const tone = value >= 0.95 ? "text-ok" : value >= 0.8 ? "text-brand" : "text-review"
  const a = 0.8 * 2 * Math.PI // angle of the threshold, measured in the rotated SVG frame
  const big = size >= 100
  return (
    <div className="relative grid shrink-0 place-items-center" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="size-full -rotate-90" aria-hidden>
        <circle cx="50" cy="50" r={r} fill="none" strokeWidth="8" className="stroke-muted" />
        <circle
          cx="50" cy="50" r={r} fill="none" strokeWidth="8" strokeLinecap="round"
          className={cn("stroke-current transition-all duration-700", tone)}
          strokeDasharray={c} strokeDashoffset={c * (1 - value)}
        />
        <line
          x1={50 + (r - 7) * Math.cos(a)} y1={50 + (r - 7) * Math.sin(a)}
          x2={50 + (r + 7) * Math.cos(a)} y2={50 + (r + 7) * Math.sin(a)}
          strokeWidth="2" strokeLinecap="round" className="stroke-foreground/50"
        >
          <title>review threshold 80%</title>
        </line>
      </svg>
      <div className="absolute text-center leading-none">
        <div className={cn("tabular font-semibold tracking-tight", big ? "text-2xl" : "text-base")}>{pct(value)}</div>
        {big && <div className="mt-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">confidence</div>}
      </div>
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

const LEGEND: { kind: Segment["kind"]; label: string; swatch: string }[] = [
  { kind: "signal", label: "Read", swatch: "bg-brand-soft border-b-2 border-brand" },
  { kind: "filler", label: "Boilerplate", swatch: "bg-muted-foreground/25" },
  { kind: "agent", label: "Agent turn", swatch: "bg-muted-foreground/15" },
  { kind: "injection", label: "Injection", swatch: "bg-urgent-soft border-b-2 border-urgent" },
  { kind: "quoted", label: "Quoted history", swatch: "bg-muted-foreground/10" },
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
            <span className={cn("inline-block h-3 w-5 rounded-sm", l.swatch)} aria-hidden />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  )
}
