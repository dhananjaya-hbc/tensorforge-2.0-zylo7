import { useState } from "react"
import { Eye, EyeOff, Loader2, Lock } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ApiError, api, getKey, setKey } from "@/lib/api"

type Props = { open: boolean; onOpenChange: (open: boolean) => void; onChange: (hasKey: boolean) => void }

export function ApiKeyDialog({ open, onOpenChange, onChange }: Props) {
  const [value, setValue] = useState("")
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const hasKey = Boolean(getKey())

  const save = async () => {
    const candidate = value.trim()
    if (!candidate) return
    setBusy(true)
    setError("")
    const previous = getKey()
    setKey(candidate)
    try {
      await api.predict({ channel: "chat", text: "connection check" })
      onChange(true)
      setValue("")
      onOpenChange(false)
      toast.success("API key verified", { description: "Stored for this browser tab only." })
    } catch (e) {
      setKey(previous)
      setError(e instanceof ApiError && e.status === 401 ? "That key was rejected (401)." : "Could not reach the API.")
    } finally {
      setBusy(false)
    }
  }

  const forget = () => {
    setKey("")
    onChange(false)
    onOpenChange(false)
    toast("API key removed from this tab")
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>API key</DialogTitle>
          <DialogDescription>
            Prediction and job endpoints require the evaluation key. Model Lab and the health status are public.
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault()
            void save()
          }}
        >
          <Label htmlFor="api-key">{hasKey ? "Replace key" : "Key"}</Label>
          <div className="relative">
            <Input
              id="api-key"
              type={show ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              placeholder="tf2_…"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="pr-10 font-mono"
            />
            <button
              type="button"
              onClick={() => setShow((s) => !s)}
              className="absolute inset-y-0 right-0 grid w-10 place-items-center text-muted-foreground hover:text-foreground"
              aria-label={show ? "Hide key" : "Show key"}
            >
              {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
          {error && <p className="text-sm text-urgent">{error}</p>}
          <p className="flex items-start gap-2 text-xs text-muted-foreground">
            <Lock className="mt-0.5 size-3.5 shrink-0" />
            Kept in this tab's session storage and sent only to this site. It is erased when the tab closes.
          </p>
          <DialogFooter className="gap-2 pt-2 sm:justify-between">
            {hasKey ? (
              <Button type="button" variant="ghost" onClick={forget}>
                Forget key
              </Button>
            ) : (
              <span />
            )}
            <Button type="submit" disabled={busy || !value.trim()}>
              {busy && <Loader2 className="size-4 animate-spin" />}
              Verify & save
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
