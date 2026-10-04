import {
  Backpack, Ban, Car, Clock, CreditCard, MessageCircleQuestion, PackageX, ShieldAlert, Smartphone, TicketPercent,
  UtensilsCrossed, type LucideIcon,
} from "lucide-react"

// Each category has its own soft colour. Colour is decoration only: the icon and label always carry the meaning.
export type CategoryInfo = { id: string; label: string; team: string; icon: LucideIcon; chip: string; dot: string }

export const CATEGORIES: CategoryInfo[] = [
  { id: "safety_conduct", label: "Safety & conduct", team: "Trust & Safety", icon: ShieldAlert,
    chip: "bg-rose-50 text-rose-700 ring-rose-200 dark:bg-rose-500/10 dark:text-rose-300 dark:ring-rose-500/25", dot: "bg-rose-500" },
  { id: "payment_refund", label: "Payment & refund", team: "Payments & Refunds", icon: CreditCard,
    chip: "bg-emerald-50 text-emerald-700 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-500/25", dot: "bg-emerald-500" },
  { id: "ride_trip_issue", label: "Ride / trip issue", team: "Ride Operations", icon: Car,
    chip: "bg-sky-50 text-sky-700 ring-sky-200 dark:bg-sky-500/10 dark:text-sky-300 dark:ring-sky-500/25", dot: "bg-sky-500" },
  { id: "lost_item", label: "Lost item", team: "Lost & Found", icon: Backpack,
    chip: "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/25", dot: "bg-amber-500" },
  { id: "order_missing_wrong", label: "Order missing / wrong", team: "Food Operations", icon: PackageX,
    chip: "bg-orange-50 text-orange-700 ring-orange-200 dark:bg-orange-500/10 dark:text-orange-300 dark:ring-orange-500/25", dot: "bg-orange-500" },
  { id: "delivery_delay", label: "Delivery delay", team: "Delivery Operations", icon: Clock,
    chip: "bg-violet-50 text-violet-700 ring-violet-200 dark:bg-violet-500/10 dark:text-violet-300 dark:ring-violet-500/25", dot: "bg-violet-500" },
  { id: "food_quality", label: "Food quality", team: "Restaurant Quality", icon: UtensilsCrossed,
    chip: "bg-lime-50 text-lime-700 ring-lime-200 dark:bg-lime-500/10 dark:text-lime-300 dark:ring-lime-500/25", dot: "bg-lime-500" },
  { id: "account_promo", label: "Account & promo", team: "Account Services", icon: TicketPercent,
    chip: "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200 dark:bg-fuchsia-500/10 dark:text-fuchsia-300 dark:ring-fuchsia-500/25", dot: "bg-fuchsia-500" },
  { id: "app_technical", label: "App / technical", team: "Tech Support", icon: Smartphone,
    chip: "bg-cyan-50 text-cyan-700 ring-cyan-200 dark:bg-cyan-500/10 dark:text-cyan-300 dark:ring-cyan-500/25", dot: "bg-cyan-500" },
  { id: "general_inquiry", label: "General inquiry", team: "Front-line Support", icon: MessageCircleQuestion,
    chip: "bg-blue-50 text-blue-700 ring-blue-200 dark:bg-blue-500/10 dark:text-blue-300 dark:ring-blue-500/25", dot: "bg-blue-500" },
  { id: "spam_irrelevant", label: "Spam / irrelevant", team: "Auto-close / Spam Filter", icon: Ban,
    chip: "bg-zinc-100 text-zinc-600 ring-zinc-200 dark:bg-zinc-500/10 dark:text-zinc-300 dark:ring-zinc-500/25", dot: "bg-zinc-400" },
]

export const CATEGORY = Object.fromEntries(CATEGORIES.map((c) => [c.id, c])) as Record<string, CategoryInfo>

export const CHANNELS = [
  { id: "chat", label: "Chat" },
  { id: "email", label: "Email" },
  { id: "call_transcript", label: "Call" },
] as const
