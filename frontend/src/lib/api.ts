// Thin client for the Sentinel API. The API key lives only in this browser tab (sessionStorage).
export type Channel = "email" | "chat" | "call_transcript"
export type Ticket = { ticket_id?: string; channel: Channel; subject?: string; text: string }
export type Segment = { text: string; kind: "signal" | "filler" | "agent" | "speaker" | "injection" | "quoted" | "space" }
export type Prediction = {
  ticket_id?: string
  category: string
  secondary_category: string | null
  team: string
  is_urgent: boolean
  confidence: number
  model_version: string
  needs_human_review?: boolean
  explain?: { segments: Segment[]; model_input: string }
}
export type JobStatus = {
  job_id: string
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled"
  total: number
  processed: number
  created_at?: string
  started_at?: string | null
  finished_at?: string | null
  expires_at?: string | null
  model_version?: string
  error?: { code: string; message: string } | null
}
export type Health = { status: "ok" | "loading"; model_version: string | null; model_loaded?: boolean }

const KEY_STORE = "sentinel.apiKey"

export function getKey(): string {
  try {
    return sessionStorage.getItem(KEY_STORE) ?? ""
  } catch {
    return ""
  }
}

export function setKey(key: string) {
  try {
    if (key) sessionStorage.setItem(KEY_STORE, key)
    else sessionStorage.removeItem(KEY_STORE)
  } catch {
    /* storage unavailable: key is kept for this page view only */
  }
  memoryKey = key
}

let memoryKey = ""
const currentKey = () => getKey() || memoryKey

export class ApiError extends Error {
  status: number
  code: string
  details: { index?: number; field?: string; issue?: string }[]
  constructor(status: number, code: string, message: string, details: ApiError["details"] = []) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

async function call<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set("Content-Type", "application/json")
  if (auth) headers.set("X-API-Key", currentKey())
  const res = await fetch(path, { ...init, headers })
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const e = body?.error ?? {}
    throw new ApiError(res.status, e.code ?? "error", e.message ?? res.statusText, e.details ?? [])
  }
  return body as T
}

export const api = {
  health: () => call<Health>("/health", {}, false),
  modelCard: () => call<Record<string, unknown>>("/model-card", {}, false),
  predict: (t: Ticket, explain = false) =>
    call<Prediction>(`/predict${explain ? "?explain=true" : ""}`, { method: "POST", body: JSON.stringify(t) }),
  predictBatch: (tickets: Ticket[]) =>
    call<{ predictions: Prediction[]; meta?: { processing_time_ms?: number } }>("/predict/batch", {
      method: "POST",
      body: JSON.stringify({ tickets }),
    }),
  submitJob: (tickets: Ticket[]) => call<JobStatus>("/batch/jobs", { method: "POST", body: JSON.stringify({ tickets }) }),
  job: (id: string) => call<JobStatus>(`/batch/jobs/${encodeURIComponent(id)}`),
  jobResults: (id: string, offset = 0, limit = 5000) =>
    call<{ predictions: Prediction[]; next_offset: number | null; total: number }>(
      `/batch/jobs/${encodeURIComponent(id)}/results?offset=${offset}&limit=${limit}`,
    ),
  deleteJob: (id: string) => call<void>(`/batch/jobs/${encodeURIComponent(id)}`, { method: "DELETE" }),
}
