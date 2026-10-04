import { useEffect, useState } from "react"
import {
  ArrowRight, Boxes, BrainCircuit, Cpu, Eraser, Gauge, ListChecks, ScanSearch, ShieldCheck, Timer, TriangleAlert, Waypoints,
} from "lucide-react"
import { Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig,
} from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api } from "@/lib/api"
import { CATEGORY } from "@/lib/labels"
import { cn, pct } from "@/lib/utils"
import lab from "@/data/modelLab.json"
import { PageTitle } from "@/pages/triage"

type Card = {
  model_version?: string
  encoder?: string
  epochs?: number
  trained_on?: string
  created_at?: string
  code_commit?: string
  onnx_parity?: { precision?: string; argmax_agreement?: number; max_abs_prob_diff?: number }
  validation_estimate?: { primary_macro_f1?: number; secondary_f1?: number; urgent_f1?: number; composite?: number }
}

const fmt = (x: number) => x.toFixed(3)
const shipped = lab.bakeoff[0]

export default function ModelLabPage() {
  const [card, setCard] = useState<Card | null>(null)
  useEffect(() => {
    api.modelCard().then((c) => setCard(c as Card)).catch(() => setCard(null))
  }, [])

  return (
    <div className="space-y-8">
      <PageTitle title="Model Lab" />
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Hero icon={<Gauge className="size-4" />} value={fmt(shipped.macro_f1)} label="macro-F1 on unseen phrasings" hint="trained on train, scored on validation" />
        <Hero icon={<ListChecks className="size-4" />} value={`${lab.harness.checks_passed}/${lab.harness.checks_total}`} label="judge-simulator checks" hint="on the live HTTPS server" />
        <Hero icon={<Timer className="size-4" />} value={`${lab.harness.job_minutes} min`} label={`for a ${lab.harness.job_tickets.toLocaleString()}-ticket job`} hint="spec target: 15 min" />
        <Hero
          icon={<ShieldCheck className="size-4" />}
          value={card?.onnx_parity?.argmax_agreement !== undefined ? pct(card.onnx_parity.argmax_agreement) : "100%"}
          label="ONNX ↔ PyTorch agreement"
          hint="exported model gives identical answers"
        />
      </section>

      <Pipeline />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">How the model improved</CardTitle>
            <CardDescription>Macro-F1 on validation (unseen phrasings)</CardDescription>
          </CardHeader>
          <CardContent>
            <Timeline />
          </CardContent>
        </Card>
        <Card className="border-review/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <TriangleAlert className="size-4 text-review" /> The trap we avoided
            </CardTitle>
            <CardDescription>Validation uses phrasings train never had</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <TrapBar label="Cross-validation inside train" value={lab.trap.cv_macro_f1} tone="bg-muted-foreground/50" note="looks perfect: memorised templates" />
            <TrapBar label="Train → validation (honest)" value={lab.trap.validation_macro_f1} tone="bg-urgent" note="same model on new phrasings" />
            <p className="text-sm text-muted-foreground">Every choice was made on train → validation.</p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Encoder bake-off</CardTitle>
            <CardDescription>4 encoders · same heads · T4 GPU · best epoch by composite</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <BakeoffTable />
            <Curves />
          </CardContent>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Per language</CardTitle>
              <CardDescription>Macro-F1 · language is never sent to the API</CardDescription>
            </CardHeader>
            <CardContent>
              <Languages />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Confidence you can trust</CardTitle>
              <CardDescription>
                Below {pct(lab.calibration.review_threshold)} → human review ({pct(lab.calibration.review_share, 1)} of tickets); the rest are{" "}
                {pct(lab.calibration.auto_accuracy, 1)} accurate
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Calibration />
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Most common mistakes</CardTitle>
            <CardDescription>Validation, LaBSE</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {lab.confusions.map((c) => (
              <div key={`${c.gold}-${c.predicted}`} className="flex items-center gap-2 text-sm">
                <span className="truncate">{CATEGORY[c.gold]?.label}</span>
                <ArrowRight className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate text-muted-foreground">{CATEGORY[c.predicted]?.label}</span>
                <Badge variant="secondary" className="tabular ml-auto">
                  {c.count}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Data</CardTitle>
            <CardDescription>Organiser dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <Row k="Train / validation" v={`${lab.dataset.train.toLocaleString()} / ${lab.dataset.validation.toLocaleString()}`} />
            <Row k="Categories" v={lab.dataset.categories} />
            <Row k="Tickets with a second issue" v={pct(lab.dataset.secondary_share)} />
            <Row k="Urgent tickets" v={pct(lab.dataset.urgent_share)} />
            <div className="flex flex-wrap gap-1.5 pt-1">
              {Object.entries(lab.dataset.languages).map(([lang, n]) => (
                <Badge key={lang} variant="outline" className="tabular">
                  {lang} {n}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model card (live)</CardTitle>
            <CardDescription>From /model-card</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <Row k="Version" v={<span className="font-mono text-xs">{card?.model_version ?? "—"}</span>} />
            <Row k="Encoder" v={card?.encoder?.split("/").pop() ?? "—"} />
            <Row k="Trained on" v={card?.trained_on ?? "—"} />
            <Row k="Epochs" v={card?.epochs ?? "—"} />
            <Row k="Runtime precision" v={card?.onnx_parity?.precision ?? "—"} />
            <Row k="Code commit" v={<span className="font-mono text-xs">{card?.code_commit ?? "—"}</span>} />
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

function Hero({ icon, value, label, hint }: { icon: React.ReactNode; value: string; label: string; hint: string }) {
  return (
    <Card className="gap-1 py-5">
      <CardContent className="space-y-1 px-5">
        <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <span className="text-brand">{icon}</span> {label}
        </div>
        <div className="tabular text-3xl font-semibold tracking-tight">{value}</div>
        <div className="text-xs text-muted-foreground">{hint}</div>
      </CardContent>
    </Card>
  )
}

const STEPS = [
  { icon: ShieldCheck, title: "Gatekeeper", text: "API key · 415 · 413 · JSON · 422" },
  { icon: Eraser, title: "Denoise", text: "boilerplate, agent turns, injections, quotes" },
  { icon: BrainCircuit, title: "LaBSE encoder", text: "109 languages, fine-tuned, 4 heads" },
  { icon: Cpu, title: "ONNX Runtime", text: "CPU, no PyTorch, padding-free batches" },
  { icon: Waypoints, title: "Decide", text: "thresholds + routing policy + review flag" },
  { icon: Boxes, title: "Response", text: "category · team · urgency · confidence" },
]

function Pipeline() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <ScanSearch className="size-4 text-brand" /> What happens to a ticket
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {STEPS.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="relative rounded-xl border bg-muted/30 p-3">
              <div className="flex items-center gap-2">
                <span className="grid size-7 place-items-center rounded-lg bg-brand-soft text-brand">
                  <Icon className="size-4" />
                </span>
                <span className="text-sm font-medium">{title}</span>
                <span className="tabular ml-auto text-xs text-muted-foreground">{i + 1}</span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{text}</p>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}

const timelineConfig = { macro_f1: { label: "Macro-F1", color: "var(--chart-1)" } } satisfies ChartConfig

function Timeline() {
  const data = lab.timeline.map((t, i) => ({ ...t, macro_f1: Number(t.macro_f1.toFixed(3)), last: i === lab.timeline.length - 1 }))
  return (
    <ChartContainer config={timelineConfig} className="aspect-auto h-[320px] w-full">
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 48 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={[0.5, 1]} ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]} tickFormatter={(v) => v.toFixed(1)} tickLine={false} axisLine={false} />
        <YAxis dataKey="step" type="category" width={236} tickLine={false} axisLine={false} tick={{ fontSize: 12 }} interval={0} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent labelKey="step" formatter={(v, _n, item) => (
          <span>
            {Number(v).toFixed(3)} <span className="text-muted-foreground">· {item.payload.detail}</span>
          </span>
        )} />} />
        <Bar dataKey="macro_f1" fill="var(--color-macro_f1)" radius={4} barSize={22}>
          <LabelList dataKey="macro_f1" position="right" className="fill-foreground" fontSize={12} formatter={(v) => Number(v).toFixed(3)} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

function TrapBar({ label, value, tone, note }: { label: string; value: number; tone: string; note: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium">{label}</span>
        <span className="tabular text-xl font-semibold">{fmt(value)}</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", tone)} style={{ width: pct(value, 1) }} />
      </div>
      <div className="text-xs text-muted-foreground">{note}</div>
    </div>
  )
}

function BakeoffTable() {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Encoder</TableHead>
          <TableHead className="text-right">Epoch</TableHead>
          <TableHead className="text-right">Macro-F1</TableHead>
          <TableHead className="text-right">Secondary F1</TableHead>
          <TableHead className="text-right">Urgent F1</TableHead>
          <TableHead className="text-right">Composite</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody className="tabular">
        {lab.bakeoff.map((b, i) => (
          <TableRow key={b.encoder} className={i === 0 ? "bg-brand-soft hover:bg-brand-soft" : ""}>
            <TableCell className="font-medium">
              {b.encoder} {i === 0 && <Badge className="ml-1 bg-brand text-brand-foreground">shipped</Badge>}
            </TableCell>
            <TableCell className="text-right">{b.epoch}</TableCell>
            <TableCell className="text-right">{fmt(b.macro_f1)}</TableCell>
            <TableCell className="text-right">{fmt(b.secondary_f1)}</TableCell>
            <TableCell className="text-right">{fmt(b.urgent_f1)}</TableCell>
            <TableCell className="text-right font-semibold">{fmt(b.composite)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

const curveConfig = Object.fromEntries(
  lab.bakeoff.map((b, i) => [`s${i}`, { label: b.encoder, color: `var(--chart-${i + 1})` }]),
) satisfies ChartConfig

function Curves() {
  const data = [1, 2, 3, 4, 5].map((epoch) =>
    Object.fromEntries([["epoch", epoch], ...lab.bakeoff.map((b, i) => [`s${i}`, b.curve.find((c) => c.epoch === epoch)?.composite ?? null])]),
  )
  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted-foreground">Composite score per epoch</div>
      <ChartContainer config={curveConfig} className="aspect-auto h-[240px] w-full">
        <LineChart data={data} margin={{ left: 4, right: 16, top: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="epoch" tickLine={false} axisLine={false} tickFormatter={(v) => `epoch ${v}`} />
          <YAxis domain={[0.55, 1]} tickLine={false} axisLine={false} width={36} tickFormatter={(v) => v.toFixed(2)} />
          <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => `Epoch ${p?.[0]?.payload?.epoch}`} formatter={(v, name) => (
            <span className="flex w-full justify-between gap-4">
              <span className="text-muted-foreground">{curveConfig[name as keyof typeof curveConfig]?.label}</span>
              <span className="tabular font-medium">{Number(v).toFixed(3)}</span>
            </span>
          )} />} />
          <ChartLegend content={<ChartLegendContent />} />
          {lab.bakeoff.map((_, i) => (
            <Line key={i} dataKey={`s${i}`} type="monotone" stroke={`var(--color-s${i})`} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          ))}
        </LineChart>
      </ChartContainer>
    </div>
  )
}

const langConfig = { macro_f1: { label: "Macro-F1", color: "var(--chart-1)" } } satisfies ChartConfig

function Languages() {
  const data = [...lab.by_language].sort((a, b) => b.n - a.n).map((l) => ({ ...l, label: `${l.language}` }))
  return (
    <ChartContainer config={langConfig} className="aspect-auto h-[220px] w-full">
      <BarChart data={data} margin={{ top: 20, left: 0, right: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 12 }} />
        <YAxis domain={[0, 1]} hide />
        <ChartTooltip cursor={false} content={<ChartTooltipContent formatter={(v, _n, item) => (
          <span>
            {Number(v).toFixed(3)} <span className="text-muted-foreground">· {item.payload.n} tickets</span>
          </span>
        )} />} />
        <Bar dataKey="macro_f1" fill="var(--color-macro_f1)" radius={4}>
          <LabelList dataKey="macro_f1" position="top" className="fill-foreground" fontSize={11} formatter={(v) => Number(v).toFixed(2)} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

const calConfig = { accuracy: { label: "Accuracy", color: "var(--chart-1)" } } satisfies ChartConfig

function Calibration() {
  return (
    <ChartContainer config={calConfig} className="aspect-auto h-[220px] w-full">
      <BarChart data={lab.calibration.bins} margin={{ top: 20, left: 0, right: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="band" tickLine={false} axisLine={false} tick={{ fontSize: 12 }} />
        <YAxis domain={[0, 1]} hide />
        <ReferenceLine y={0.95} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
        <ChartTooltip cursor={false} content={<ChartTooltipContent formatter={(v, _n, item) => (
          <span>
            {pct(Number(v), 1)} accurate <span className="text-muted-foreground">· {item.payload.tickets} tickets</span>
          </span>
        )} />} />
        <Bar dataKey="accuracy" fill="var(--color-accuracy)" radius={4}>
          <LabelList dataKey="accuracy" position="top" className="fill-foreground" fontSize={11} formatter={(v) => pct(Number(v))} />
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted-foreground">{k}</span>
      <span className="truncate text-right">{v}</span>
    </div>
  )
}
