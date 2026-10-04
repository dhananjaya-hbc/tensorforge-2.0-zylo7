import type { Channel, Prediction, Ticket } from "@/lib/api"

export type LabeledTicket = Ticket & { ticket_id: string; gold?: { category?: string; is_urgent?: boolean } }

const CHANNELS = new Set(["email", "chat", "call_transcript"])

/** RFC 4180 CSV parser: quoted fields may contain commas, quotes ("") and newlines. */
export function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"') quoted = true
    else if (ch === ",") {
      row.push(field)
      field = ""
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++
      row.push(field)
      rows.push(row)
      row = []
      field = ""
    } else field += ch
  }
  if (field || row.length) {
    row.push(field)
    rows.push(row)
  }
  const [header, ...body] = rows.filter((r) => r.some((c) => c.trim()))
  if (!header) return []
  const keys = header.map((h) => h.trim().replace(/^﻿/, ""))
  return body.map((r) => Object.fromEntries(keys.map((k, i) => [k, r[i] ?? ""])))
}

function toTicket(raw: Record<string, unknown>, i: number): LabeledTicket {
  const urgent = raw.is_urgent
  return {
    ticket_id: String(raw.ticket_id ?? "").trim() || `ROW-${String(i + 1).padStart(5, "0")}`,
    channel: String(raw.channel ?? "").trim() as Channel,
    subject: raw.subject == null ? "" : String(raw.subject),
    text: raw.text == null ? "" : String(raw.text),
    gold: raw.category
      ? { category: String(raw.category), is_urgent: urgent === true || String(urgent).toLowerCase() === "true" }
      : undefined,
  }
}

export async function readTickets(file: File): Promise<LabeledTicket[]> {
  const text = await file.text()
  const name = file.name.toLowerCase()
  let records: Record<string, unknown>[]
  if (name.endsWith(".csv")) records = parseCsv(text)
  else if (name.endsWith(".jsonl") || name.endsWith(".ndjson"))
    records = text.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l))
  else {
    const data = JSON.parse(text)
    records = Array.isArray(data) ? data : data.tickets
  }
  if (!Array.isArray(records)) throw new Error("Expected a list of tickets")
  return records.map(toTicket)
}

export function problems(tickets: LabeledTicket[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  tickets.forEach((t, i) => {
    if (!CHANNELS.has(t.channel)) out.push(`row ${i + 1}: channel must be email, chat or call_transcript`)
    if (!t.text.trim()) out.push(`row ${i + 1}: text is empty`)
    if (t.text.length > 10000) out.push(`row ${i + 1}: text longer than 10,000 characters`)
    if (seen.has(t.ticket_id)) out.push(`row ${i + 1}: duplicate ticket_id ${t.ticket_id}`)
    seen.add(t.ticket_id)
  })
  if (tickets.length > 5000) out.push("a job holds at most 5,000 tickets")
  return out
}

function csvCell(v: unknown) {
  const s = v == null ? "" : String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const a = Object.assign(document.createElement("a"), { href: url, download: name })
  a.click()
  URL.revokeObjectURL(url)
}

export function predictionsCsv(preds: Prediction[]) {
  const cols = ["ticket_id", "category", "secondary_category", "team", "is_urgent", "confidence", "needs_human_review", "model_version"] as const
  return [cols.join(","), ...preds.map((p) => cols.map((c) => csvCell(p[c])).join(","))].join("\n")
}
