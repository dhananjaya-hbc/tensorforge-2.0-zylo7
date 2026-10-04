import { useEffect, useMemo, useRef, useState } from "react"
import { AlertTriangle, CheckCircle2, Download, FileUp, Loader2, Play, Siren, Square, UserRoundSearch, XCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { CategoryMark, reportError } from "@/components/prediction-parts"
import { api, type JobStatus, type Prediction } from "@/lib/api"
import { download, predictionsCsv, problems, readTickets, type LabeledTicket } from "@/lib/files"
import { CATEGORY, CATEGORIES } from "@/lib/labels"
import { cn, pct } from "@/lib/utils"
import samples from "@/data/samples.json"
import { PageTitle } from "@/pages/triage"

const DEMO: LabeledTicket[] = (samples.stream as (LabeledTicket & { language: string })[]).map(
  ({ language: _language, ...t }) => t,
)
const PREVIEW = 15

export default function BatchLabPage() {
  const [tickets, setTickets] = useState<LabeledTicket[] | null>(null)
  const [source, setSource] = useState("")
  const [job, setJob] = useState<JobStatus | null>(null)
  const [results, setResults] = useState<Prediction[] | null>(null)
  const [startedAt, setStartedAt] = useState(0)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const issues = useMemo(() => (tickets ? problems(tickets) : []), [tickets])
  const active = job && (job.status === "queued" || job.status === "running")

  const load = async (file: File) => {
    try {
      setTickets(await readTickets(file))
      setSource(file.name)
      setJob(null)
      setResults(null)
    } catch (e) {
      reportError(new Error(`Could not read ${file.name}: ${e instanceof Error ? e.message : e}`))
    }
  }

  const submit = async () => {
    if (!tickets) return
    try {
      setResults(null)
      setStartedAt(Date.now())
      setJob(await api.submitJob(tickets.map(({ ticket_id, channel, subject, text }) => ({ ticket_id, channel, subject, text }))))
    } catch (e) {
      reportError(e)
    }
  }

  useEffect(() => {
    if (!job || !active) return
    const id = window.setTimeout(async () => {
      try {
        const next = await api.job(job.job_id)
        setJob(next)
        if (next.status === "succeeded") {
          const all: Prediction[] = []
          let offset: number | null = 0
          while (offset !== null) {
            const page: { predictions: Prediction[]; next_offset: number | null } = await api.jobResults(job.job_id, offset, 5000)
            all.push(...page.predictions)
            offset = page.next_offset
          }
          setResults(all)
        }
      } catch (e) {
        reportError(e)
      }
    }, 1500)
    return () => window.clearTimeout(id)
  }, [job, active])

  const cancel = async () => {
    if (!job) return
    try {
      await api.deleteJob(job.job_id)
      setJob({ ...job, status: "cancelled" })
    } catch (e) {
      reportError(e)
    }
  }

  return (
    <div className="space-y-6">
      <PageTitle title="Batch" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Load tickets</CardTitle>
            <CardDescription>CSV, JSONL or JSON · up to 5,000 tickets · labels optional</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <button
              onClick={() => input.current?.click()}
              onDragOver={(e) => (e.preventDefault(), setDragging(true))}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                const f = e.dataTransfer.files[0]
                if (f) void load(f)
              }}
              className={cn(
                "flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors hover:border-brand/50 hover:bg-brand-soft",
                dragging && "border-brand bg-brand-soft",
              )}
            >
              <FileUp className="size-7 text-brand" />
              <span className="text-sm font-medium">Drop a file or click to browse</span>
              <span className="text-xs text-muted-foreground">.csv · .jsonl · .json</span>
            </button>
            <input ref={input} type="file" accept=".csv,.jsonl,.ndjson,.json" className="hidden" onChange={(e) => e.target.files?.[0] && load(e.target.files[0])} />
            <div className="flex items-center gap-3 text-sm">
              <div className="h-px flex-1 bg-border" />
              <span className="text-muted-foreground">or</span>
              <div className="h-px flex-1 bg-border" />
            </div>
            <Button variant="outline" className="w-full" onClick={() => (setTickets(DEMO), setSource("demo set"), setJob(null), setResults(null))}>
              Use {DEMO.length} labelled demo tickets
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run</CardTitle>
            <CardDescription>Async job: submit → poll → results</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!tickets ? (
              <div className="grid h-40 place-items-center rounded-xl border border-dashed text-sm text-muted-foreground">Load tickets to begin</div>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge variant="secondary" className="tabular">
                    {tickets.length.toLocaleString()} tickets
                  </Badge>
                  <span className="truncate text-muted-foreground">from {source}</span>
                  {tickets.some((t) => t.gold) && <Badge variant="outline">labelled</Badge>}
                </div>
                {issues.length > 0 && (
                  <Alert variant="destructive">
                    <AlertTriangle className="size-4" />
                    <AlertTitle>{issues.length} problem(s); the API would reject this job with 422</AlertTitle>
                    <AlertDescription className="max-h-24 overflow-y-auto font-mono text-xs">{issues.slice(0, 8).join("\n")}</AlertDescription>
                  </Alert>
                )}
                <div className="flex gap-2">
                  <Button onClick={submit} disabled={Boolean(active)}>
                    {active ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                    Submit job
                  </Button>
                  {active && (
                    <Button variant="outline" onClick={cancel}>
                      <Square className="size-4" /> Cancel
                    </Button>
                  )}
                </div>
                {job && <JobProgress job={job} startedAt={startedAt} />}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {results && tickets && <Results preds={results} tickets={tickets} />}
    </div>
  )
}

function JobProgress({ job, startedAt }: { job: JobStatus; startedAt: number }) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 500)
    return () => window.clearInterval(id)
  }, [])
  const share = job.total ? job.processed / job.total : 0
  const seconds = Math.max(1, (now - startedAt) / 1000)
  const eta = share > 0.02 && share < 1 ? Math.round(seconds / share - seconds) : null
  const icon = {
    queued: <Loader2 className="size-4 animate-spin text-muted-foreground" />,
    running: <Loader2 className="size-4 animate-spin text-brand" />,
    succeeded: <CheckCircle2 className="size-4 text-ok" />,
    failed: <XCircle className="size-4 text-urgent" />,
    cancelled: <XCircle className="size-4 text-muted-foreground" />,
  }[job.status]
  return (
    <div className="space-y-2 rounded-xl border bg-muted/40 p-4">
      <div className="flex items-center gap-2 text-sm font-medium capitalize">
        {icon} {job.status}
        <span className="ml-auto tabular text-muted-foreground">
          {job.processed.toLocaleString()} / {job.total.toLocaleString()}
        </span>
      </div>
      <Progress value={share * 100} className="h-2" />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span className="font-mono">{job.job_id}</span>
        <span className="tabular">{job.status === "succeeded" ? `done in ${seconds.toFixed(0)} s` : eta !== null ? `~${eta} s left` : ""}</span>
      </div>
      {job.error && <p className="text-sm text-urgent">{job.error.message}</p>}
    </div>
  )
}

function Results({ preds, tickets }: { preds: Prediction[]; tickets: LabeledTicket[] }) {
  const [all, setAll] = useState(false)
  const shown = all ? preds.slice(0, 1000) : preds.slice(0, PREVIEW)
  const byId = new Map(tickets.map((t) => [t.ticket_id, t]))
  const labelled = preds.filter((p) => byId.get(p.ticket_id ?? "")?.gold?.category)
  const correct = labelled.filter((p) => byId.get(p.ticket_id ?? "")!.gold!.category === p.category).length
  const counts = CATEGORIES.map((c) => ({ ...c, n: preds.filter((p) => p.category === c.id).length })).filter((c) => c.n)
  const max = Math.max(...counts.map((c) => c.n), 1)
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-3 gap-3 text-center">
            <Mini label="Urgent" value={preds.filter((p) => p.is_urgent).length} icon={<Siren className="size-3.5 text-urgent" />} />
            <Mini label="Review" value={preds.filter((p) => p.needs_human_review).length} icon={<UserRoundSearch className="size-3.5 text-review" />} />
            <Mini label="Accuracy" value={labelled.length ? pct(correct / labelled.length) : "—"} icon={<CheckCircle2 className="size-3.5 text-ok" />} />
          </div>
          <div className="space-y-2">
            {counts.map((c) => (
              <div key={c.id} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 text-xs">
                <span className="truncate">{c.team}</span>
                <span className="tabular text-muted-foreground">{c.n}</span>
                <div className="col-span-2 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-brand" style={{ width: `${(c.n / max) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => download(`sentinel-results-${stamp}.csv`, predictionsCsv(preds), "text/csv")}>
              <Download className="size-4" /> CSV
            </Button>
            <Button variant="outline" size="sm" onClick={() => download(`sentinel-results-${stamp}.json`, JSON.stringify({ predictions: preds }, null, 2), "application/json")}>
              <Download className="size-4" /> JSON
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Predictions</CardTitle>
          <CardDescription>
            In request order · {shown.length} of {preds.length.toLocaleString()} shown
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ticket</TableHead>
                <TableHead>Category → team</TableHead>
                <TableHead>Flags</TableHead>
                <TableHead className="text-right">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((p) => {
                const gold = byId.get(p.ticket_id ?? "")?.gold?.category
                return (
                  <TableRow key={p.ticket_id}>
                    <TableCell className="font-mono text-xs">{p.ticket_id}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <CategoryMark category={p.category} size="sm" />
                        <div className="min-w-0">
                          <div className="truncate text-sm">{CATEGORY[p.category]?.label}</div>
                          <div className="truncate text-xs text-muted-foreground">{p.team}</div>
                        </div>
                        {gold && (gold === p.category ? <CheckCircle2 className="ml-1 size-3.5 shrink-0 text-ok" /> : <XCircle className="ml-1 size-3.5 shrink-0 text-urgent" />)}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {p.is_urgent && <Siren className="size-4 text-urgent" aria-label="urgent" />}
                        {p.needs_human_review && <UserRoundSearch className="size-4 text-review" aria-label="needs review" />}
                      </div>
                    </TableCell>
                    <TableCell className="tabular text-right">{pct(p.confidence, 1)}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          {preds.length > PREVIEW && (
            <Button variant="ghost" size="sm" className="mt-3 w-full" onClick={() => setAll((a) => !a)}>
              {all ? "Show fewer" : `Show ${Math.min(preds.length, 1000) - PREVIEW} more`}
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function Mini({ label, value, icon }: { label: string; value: React.ReactNode; icon: React.ReactNode }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="flex items-center justify-center gap-1 text-[11px] text-muted-foreground">
        {icon} {label}
      </div>
      <div className="tabular mt-1 text-lg font-semibold">{value}</div>
    </div>
  )
}
