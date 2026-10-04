import { useRef, useState } from "react"
import { ArrowRight, Braces, Eraser, Loader2, Sparkles, Timer } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { CategoryChip, ConfidenceRing, Evidence, ReviewBadge, UrgencyBadge, reportError } from "@/components/prediction-parts"
import { api, type Channel, type Prediction, type Ticket } from "@/lib/api"
import { CATEGORY, CHANNELS } from "@/lib/labels"
import { cn } from "@/lib/utils"
import samples from "@/data/samples.json"

const MAX = 10000

export function PageTitle({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
      {children}
    </div>
  )
}

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

  // A sample only fills the form; the user reviews or edits it and presses Classify.
  const useSample = (t: Ticket) => {
    setChannel(t.channel)
    setSubject(t.subject ?? "")
    setText(t.text)
    setResult(null)
    textRef.current?.focus()
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  return (
    <>
      <PageTitle title="Live Triage" />
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-5">
          <form
            className="space-y-4 rounded-2xl border bg-card p-5 shadow-sm"
            onSubmit={(e) => {
              e.preventDefault()
              void run()
            }}
          >
            <div className="flex flex-wrap items-center gap-3">
              <div className="inline-flex rounded-full border bg-muted/60 p-1" role="radiogroup" aria-label="Channel">
                {CHANNELS.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    role="radio"
                    aria-checked={channel === c.id}
                    onClick={() => setChannel(c.id)}
                    className={cn(
                      "rounded-full px-4 py-1.5 text-sm text-muted-foreground transition-all",
                      channel === c.id && "bg-card font-medium text-foreground shadow-sm",
                    )}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              <span className={cn("tabular ml-auto text-xs", text.length > MAX ? "text-urgent" : "text-muted-foreground")}>
                {text.length.toLocaleString()} / {MAX.toLocaleString()}
              </span>
            </div>
            {channel === "email" && (
              <Input placeholder="Subject" value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={500} className="animate-in fade-in" />
            )}
            <Textarea
              ref={textRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void run()
              }}
              rows={8}
              placeholder="Paste a ticket in any language…"
              className="min-h-52 resize-y border-0 bg-muted/40 text-[15px] leading-7 shadow-none focus-visible:ring-2"
            />
            <div className="flex items-center gap-2">
              <Button type="submit" className="rounded-full bg-gradient-to-r from-brand to-brand-2 px-5 shadow-md shadow-brand/20" disabled={busy || !text.trim() || text.length > MAX}>
                {busy ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
                Classify
              </Button>
              <Button type="button" variant="ghost" className="rounded-full" onClick={() => (setText(""), setSubject(""), setResult(null))}>
                <Eraser className="size-4" /> Clear
              </Button>
              <kbd className="ml-auto hidden rounded border bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground sm:block">⌘ ↵</kbd>
            </div>
          </form>

          <div className="space-y-3">
            <div className="text-sm font-medium text-muted-foreground">Try a sample</div>
            <div className="flex flex-wrap gap-2">
              {samples.showcase.map((s) => (
                <button
                  key={s.title}
                  onClick={() => useSample(s.ticket as Ticket)}
                  className="group inline-flex items-center gap-2 rounded-full border bg-card py-1.5 pr-3.5 pl-1.5 text-sm shadow-sm transition-all hover:-translate-y-px hover:border-brand/40 hover:shadow"
                >
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground group-hover:bg-brand-soft group-hover:text-brand">
                    {s.language}
                  </span>
                  {s.title.split("·").pop()?.trim()}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          {busy && !result ? <ResultSkeleton /> : result ? <Result pred={result.pred} ms={result.ms} busy={busy} /> : <EmptyState />}
        </div>
      </div>
    </>
  )
}

function EmptyState() {
  return (
    <div className="grid min-h-[26rem] place-items-center rounded-2xl border border-dashed bg-card/50">
      <div className="space-y-3 p-8 text-center">
        <div className="mx-auto grid size-14 place-items-center rounded-2xl bg-gradient-to-br from-brand to-brand-2 text-white shadow-lg shadow-brand/25">
          <Sparkles className="size-6" />
        </div>
        <div className="font-medium">Your result appears here</div>
        <div className="text-sm text-muted-foreground">Category · team · urgency · confidence · evidence</div>
      </div>
    </div>
  )
}

function ResultSkeleton() {
  return (
    <div className="space-y-5 rounded-2xl border bg-card p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="space-y-3">
          <Skeleton className="h-7 w-44 rounded-full" />
          <Skeleton className="h-5 w-36" />
        </div>
        <Skeleton className="size-28 rounded-full" />
      </div>
      <Skeleton className="h-24 w-full" />
    </div>
  )
}

function Result({ pred, ms, busy }: { pred: Prediction; ms: number; busy: boolean }) {
  const { explain, ...response } = pred
  return (
    <>
      <div className={cn("animate-in fade-in rounded-2xl border bg-card p-6 shadow-sm transition-opacity", busy && "opacity-60")}>
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0 space-y-3">
            <CategoryChip category={pred.category} className="px-3 py-1.5 text-sm" />
            <div className="flex items-center gap-2 text-xl font-semibold tracking-tight">
              <ArrowRight className="size-5 text-muted-foreground" />
              {pred.team}
            </div>
            <div className="flex flex-wrap gap-2">
              <UrgencyBadge urgent={pred.is_urgent} />
              <ReviewBadge review={pred.needs_human_review} />
            </div>
          </div>
          <ConfidenceRing value={pred.confidence} size={120} />
        </div>
        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 border-t pt-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-2">
            Also:
            {pred.secondary_category ? <CategoryChip category={pred.secondary_category} /> : <span className="text-foreground">—</span>}
          </span>
          <span className="tabular flex items-center gap-1.5">
            <Timer className="size-3.5" /> {ms} ms
          </span>
          <span className="ml-auto truncate font-mono text-xs">{pred.model_version}</span>
        </div>
      </div>

      {explain && (
        <div className="space-y-3 rounded-2xl border bg-card p-6 shadow-sm">
          <div className="text-sm font-medium">What the model read</div>
          <Evidence segments={explain.segments} />
        </div>
      )}

      <Collapsible className="rounded-2xl border bg-card shadow-sm">
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-6 py-4 text-left text-sm font-medium">
          <Braces className="size-4 text-muted-foreground" /> API response
          <span className="ml-auto text-xs font-normal text-muted-foreground">{CATEGORY[pred.category]?.label}</span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="mx-6 mb-6 overflow-x-auto rounded-xl bg-muted/60 p-4 font-mono text-xs leading-relaxed">
            {JSON.stringify(response, null, 2)}
          </pre>
          {explain && (
            <div className="mx-6 mb-6 space-y-1">
              <div className="text-xs text-muted-foreground">Exact model input</div>
              <pre className="overflow-x-auto whitespace-pre-wrap rounded-xl bg-muted/60 p-3 font-mono text-xs">{explain.model_input}</pre>
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
    </>
  )
}
