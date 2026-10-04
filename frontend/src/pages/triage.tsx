import { useRef, useState } from "react"
import { Braces, Eraser, Loader2, ScanText, Sparkles, Timer } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import {
  CategoryHeading, CategoryMark, ConfidenceMeter, Evidence, ReviewBadge, UrgencyBadge, reportError,
} from "@/components/prediction-parts"
import { api, type Channel, type Prediction, type Ticket } from "@/lib/api"
import { CATEGORY, CHANNELS } from "@/lib/labels"
import samples from "@/data/samples.json"

const MAX = 10000

export default function TriagePage() {
  const [channel, setChannel] = useState<Channel>("chat")
  const [subject, setSubject] = useState("")
  const [text, setText] = useState("")
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ pred: Prediction; ms: number } | null>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)

  const run = async (ticket?: Ticket) => {
    const t: Ticket = ticket ?? { channel, subject: channel === "email" ? subject : "", text }
    if (!t.text.trim()) return textRef.current?.focus()
    setBusy(true)
    const start = performance.now()
    try {
      const pred = await api.predict(t, true)
      setResult({ pred, ms: Math.round(performance.now() - start) })
    } catch (e) {
      reportError(e)
    } finally {
      setBusy(false)
    }
  }

  const useSample = (t: Ticket) => {
    setChannel(t.channel)
    setSubject(t.subject ?? "")
    setText(t.text)
    void run(t)
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ScanText className="size-4 text-brand" /> Ticket
            </CardTitle>
            <CardDescription>English, Sinhala, Tamil, Singlish, Tanglish or mixed. Emoji welcome.</CardDescription>
          </CardHeader>
          <CardContent>
            <form
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault()
                void run()
              }}
            >
              <div className="grid gap-4 sm:grid-cols-[10rem_1fr]">
                <div className="space-y-2">
                  <Label>Channel</Label>
                  <Select value={channel} onValueChange={(v) => setChannel(v as Channel)}>
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {CHANNELS.map((c) => (
                        <SelectItem key={c.id} value={c.id}>
                          {c.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="subject" className={channel !== "email" ? "text-muted-foreground" : ""}>
                    Subject {channel !== "email" && <span className="font-normal">(email only)</span>}
                  </Label>
                  <Input id="subject" value={subject} disabled={channel !== "email"} onChange={(e) => setSubject(e.target.value)} maxLength={500} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <Label htmlFor="text">Message</Label>
                  <span className={`tabular text-xs ${text.length > MAX ? "text-urgent" : "text-muted-foreground"}`}>
                    {text.length.toLocaleString()} / {MAX.toLocaleString()}
                  </span>
                </div>
                <Textarea
                  id="text"
                  ref={textRef}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void run()
                  }}
                  rows={9}
                  placeholder="e.g. mage order eka hour ekakata late, refund ekak denna"
                  className="min-h-48 resize-y text-[15px] leading-7"
                />
              </div>
              <div className="flex items-center gap-2">
                <Button type="submit" disabled={busy || !text.trim() || text.length > MAX}>
                  {busy ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
                  Classify
                </Button>
                <Button type="button" variant="ghost" onClick={() => (setText(""), setSubject(""), setResult(null))}>
                  <Eraser className="size-4" /> Clear
                </Button>
                <span className="ml-auto hidden text-xs text-muted-foreground sm:block">⌘/Ctrl + Enter</span>
              </div>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Try a sample</CardTitle>
            <CardDescription>The first six are the examples from the API specification.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2 sm:grid-cols-2">
            {samples.showcase.map((s) => (
              <button
                key={s.title}
                onClick={() => useSample(s.ticket as Ticket)}
                className="group flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors hover:border-brand/50 hover:bg-brand-soft"
              >
                <Badge variant="secondary" className="text-[10px]">
                  {s.language}
                </Badge>
                <span className="text-sm font-medium leading-snug">{s.title}</span>
                <span className="line-clamp-1 text-xs text-muted-foreground">{s.ticket.text}</span>
              </button>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6">
        {busy && !result ? (
          <ResultSkeleton />
        ) : result ? (
          <Result pred={result.pred} ms={result.ms} busy={busy} />
        ) : (
          <EmptyState />
        )}
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <Card className="grid min-h-[28rem] place-items-center border-dashed">
      <div className="max-w-sm space-y-3 p-8 text-center">
        <div className="mx-auto grid size-12 place-items-center rounded-xl bg-brand-soft text-brand">
          <Sparkles className="size-6" />
        </div>
        <h2 className="text-lg font-semibold">Classify a support ticket</h2>
        <p className="text-sm text-muted-foreground">
          Paste a ticket or pick a sample. You will see the category, the team it routes to, urgency, how sure the model
          is, and which words it actually read.
        </p>
      </div>
    </Card>
  )
}

function ResultSkeleton() {
  return (
    <Card>
      <CardContent className="space-y-6 pt-6">
        <div className="flex items-center gap-3">
          <Skeleton className="size-12 rounded-xl" />
          <div className="space-y-2">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-4 w-32" />
          </div>
        </div>
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-24 w-full" />
      </CardContent>
    </Card>
  )
}

function Result({ pred, ms, busy }: { pred: Prediction; ms: number; busy: boolean }) {
  const { explain, ...response } = pred
  return (
    <>
      <Card className={busy ? "opacity-60 transition-opacity" : "transition-opacity"}>
        <CardContent className="space-y-6 pt-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <CategoryHeading category={pred.category} team={pred.team} />
            <div className="flex flex-wrap gap-2">
              <UrgencyBadge urgent={pred.is_urgent} />
              <ReviewBadge review={pred.needs_human_review} />
            </div>
          </div>
          <ConfidenceMeter value={pred.confidence} />
          <Separator />
          <div className="grid gap-4 text-sm sm:grid-cols-3">
            <Fact label="Second issue">
              {pred.secondary_category ? (
                <span className="flex items-center gap-2">
                  <CategoryMark category={pred.secondary_category} size="sm" />
                  {CATEGORY[pred.secondary_category]?.label}
                </span>
              ) : (
                <span className="text-muted-foreground">None</span>
              )}
            </Fact>
            <Fact label="Response time">
              <span className="tabular flex items-center gap-1.5">
                <Timer className="size-3.5 text-muted-foreground" /> {ms} ms
              </span>
            </Fact>
            <Fact label="Model">
              <span className="truncate font-mono text-xs">{pred.model_version}</span>
            </Fact>
          </div>
        </CardContent>
      </Card>

      {explain && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">What the model read</CardTitle>
            <CardDescription>
              Boilerplate, agent lines, quoted history and prompt injections are removed before classification.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Evidence segments={explain.segments} />
            <Collapsible>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="-ml-2 text-muted-foreground">
                  Exact model input
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed">
                  {explain.model_input}
                </pre>
              </CollapsibleContent>
            </Collapsible>
          </CardContent>
        </Card>
      )}

      <Collapsible>
        <Card className="py-0">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-6 py-4 text-left text-sm font-medium">
            <Braces className="size-4 text-muted-foreground" /> API response (spec format)
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mx-6 mb-6 overflow-x-auto rounded-lg bg-muted p-4 font-mono text-xs leading-relaxed">
              {JSON.stringify(response, null, 2)}
            </pre>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </>
  )
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}
