import { useEffect, useRef, useState } from "react"
import { Check, Pause, Play, RotateCcw, Siren, UserRoundSearch, X } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CategoryMark, confidenceTone, reportError } from "@/components/prediction-parts"
import { api, type Prediction, type Ticket } from "@/lib/api"
import { CATEGORIES } from "@/lib/labels"
import { cn, pct } from "@/lib/utils"
import samples from "@/data/samples.json"

type StreamTicket = Ticket & { ticket_id: string; language: string; gold: { category: string; is_urgent: boolean } }
type Routed = { ticket: StreamTicket; pred: Prediction; at: number }

const STREAM = samples.stream as StreamTicket[]
const BATCH = 3
const PER_LANE = 4

export default function OpsBoardPage() {
  const [routed, setRouted] = useState<Routed[]>([])
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState("1")
  const cursor = useRef(0)
  const inFlight = useRef(false)

  useEffect(() => {
    if (!running) return
    const tick = async () => {
      if (inFlight.current) return
      if (cursor.current >= STREAM.length) return setRunning(false)
      inFlight.current = true
      const batch = STREAM.slice(cursor.current, cursor.current + BATCH)
      try {
        const { predictions } = await api.predictBatch(
          batch.map(({ ticket_id, channel, subject, text }) => ({ ticket_id, channel, subject, text })),
        )
        cursor.current += batch.length
        const now = Date.now()
        setRouted((r) => [...batch.map((ticket, i) => ({ ticket, pred: predictions[i], at: now + i })), ...r])
      } catch (e) {
        setRunning(false)
        reportError(e)
      } finally {
        inFlight.current = false
      }
    }
    void tick()
    const id = window.setInterval(tick, 1400 / Number(speed))
    return () => window.clearInterval(id)
  }, [running, speed])

  const reset = () => {
    setRunning(false)
    cursor.current = 0
    setRouted([])
  }

  const urgent = routed.filter((r) => r.pred.is_urgent)
  const review = routed.filter((r) => r.pred.needs_human_review)
  const correct = routed.filter((r) => r.pred.category === r.ticket.gold.category).length
  const done = routed.length

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => setRunning((r) => !r)} disabled={done >= STREAM.length && !running}>
          {running ? <Pause className="size-4" /> : <Play className="size-4" />}
          {running ? "Pause" : done ? "Resume" : "Start stream"}
        </Button>
        <Button variant="outline" onClick={reset} disabled={!done && !running}>
          <RotateCcw className="size-4" /> Reset
        </Button>
        <Select value={speed} onValueChange={setSpeed}>
          <SelectTrigger className="w-28" aria-label="Speed">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["1", "2", "4"].map((s) => (
              <SelectItem key={s} value={s}>
                {s}× speed
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-sm text-muted-foreground">
          {STREAM.length} multilingual validation tickets, sent to <code className="font-mono text-xs">/predict/batch</code> {BATCH} at a time.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Routed" value={`${done} / ${STREAM.length}`}>
          <Progress value={(done / STREAM.length) * 100} className="h-1.5" />
        </Stat>
        <Stat label="Urgent" value={urgent.length} icon={<Siren className="size-4 text-urgent" />} />
        <Stat label="Sent to human review" value={review.length} icon={<UserRoundSearch className="size-4 text-review" />} />
        <Stat label="Agree with true label" value={done ? pct(correct / done) : "—"} icon={<Check className="size-4 text-ok" />} />
      </div>

      {urgent.length > 0 && (
        <Card className="border-urgent/40 bg-urgent-soft py-0">
          <div className="flex items-center gap-3 overflow-x-auto px-4 py-3">
            <span className="flex shrink-0 items-center gap-1.5 text-sm font-semibold text-urgent">
              <Siren className="size-4" /> Urgent queue
            </span>
            {urgent.slice(0, 12).map((r) => (
              <span key={r.ticket.ticket_id} className="shrink-0 animate-in fade-in rounded-md border border-urgent/30 bg-background px-2.5 py-1 text-xs">
                <span className="font-mono text-muted-foreground">{r.ticket.ticket_id}</span> · {r.pred.team}
              </span>
            ))}
          </div>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
        {CATEGORIES.map((c) => {
          const lane = routed.filter((r) => r.pred.category === c.id)
          return (
            <Card key={c.id} className="gap-3 py-4">
              <CardHeader className="flex flex-row items-center gap-3 px-4">
                <CategoryMark category={c.id} size="sm" />
                <div className="min-w-0 flex-1">
                  <CardTitle className="truncate text-sm">{c.team}</CardTitle>
                  <div className="truncate text-xs text-muted-foreground">{c.label}</div>
                </div>
                <Badge variant="secondary" className="tabular">
                  {lane.length}
                </Badge>
              </CardHeader>
              <CardContent className="space-y-2 px-4">
                {lane.length === 0 && <div className="rounded-lg border border-dashed py-6 text-center text-xs text-muted-foreground">No tickets yet</div>}
                {lane.slice(0, PER_LANE).map((r) => (
                  <TicketCard key={r.ticket.ticket_id} r={r} />
                ))}
                {lane.length > PER_LANE && <div className="text-center text-xs text-muted-foreground">+{lane.length - PER_LANE} more</div>}
              </CardContent>
            </Card>
          )
        })}
        <Card className="gap-3 border-review/40 py-4">
          <CardHeader className="flex flex-row items-center gap-3 px-4">
            <div className="grid size-6 place-items-center rounded-md bg-review-soft text-review">
              <UserRoundSearch className="size-3.5" />
            </div>
            <div className="flex-1">
              <CardTitle className="text-sm">Human review queue</CardTitle>
              <div className="text-xs text-muted-foreground">confidence below 80%</div>
            </div>
            <Badge variant="secondary">{review.length}</Badge>
          </CardHeader>
          <CardContent className="space-y-2 px-4">
            {review.length === 0 && <div className="rounded-lg border border-dashed py-6 text-center text-xs text-muted-foreground">Nothing uncertain yet</div>}
            {review.slice(0, PER_LANE).map((r) => (
              <TicketCard key={`rv-${r.ticket.ticket_id}`} r={r} />
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Stat({ label, value, icon, children }: { label: string; value: React.ReactNode; icon?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <Card className="gap-2 py-4">
      <CardContent className="space-y-2 px-4">
        <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
          {label}
          {icon}
        </div>
        <div className="tabular text-2xl font-semibold tracking-tight">{value}</div>
        {children}
      </CardContent>
    </Card>
  )
}

function TicketCard({ r }: { r: Routed }) {
  const right = r.pred.category === r.ticket.gold.category
  return (
    <div
      className={cn(
        "animate-in fade-in slide-in-from-top-2 space-y-1.5 rounded-lg border bg-background p-2.5 duration-500",
        r.pred.is_urgent && "border-urgent/50",
      )}
    >
      <div className="flex items-center gap-1.5 text-[11px]">
        <span className="font-mono text-muted-foreground">{r.ticket.ticket_id}</span>
        <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
          {r.ticket.language}
        </Badge>
        {r.pred.is_urgent && <Siren className="size-3 text-urgent" aria-label="urgent" />}
        {r.pred.needs_human_review && <UserRoundSearch className="size-3 text-review" aria-label="needs review" />}
        <span className="ml-auto" title={right ? "matches the true label" : `true label: ${r.ticket.gold.category}`}>
          {right ? <Check className="size-3.5 text-ok" /> : <X className="size-3.5 text-urgent" />}
        </span>
      </div>
      <p className="line-clamp-2 text-xs leading-relaxed">{r.ticket.text}</p>
      <div className="flex items-center gap-2">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
          <div className={cn("h-full rounded-full", confidenceTone(r.pred.confidence))} style={{ width: pct(r.pred.confidence) }} />
        </div>
        <span className="tabular text-[10px] text-muted-foreground">{pct(r.pred.confidence)}</span>
      </div>
    </div>
  )
}
