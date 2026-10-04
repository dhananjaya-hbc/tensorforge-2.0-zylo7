import {
  Backpack, Ban, Car, Clock, CreditCard, MessageCircleQuestion, PackageX, ShieldAlert, Smartphone, TicketPercent,
  UtensilsCrossed, type LucideIcon,
} from "lucide-react"

export type CategoryInfo = { id: string; label: string; team: string; icon: LucideIcon }

export const CATEGORIES: CategoryInfo[] = [
  { id: "safety_conduct", label: "Safety & conduct", team: "Trust & Safety", icon: ShieldAlert },
  { id: "payment_refund", label: "Payment & refund", team: "Payments & Refunds", icon: CreditCard },
  { id: "ride_trip_issue", label: "Ride / trip issue", team: "Ride Operations", icon: Car },
  { id: "lost_item", label: "Lost item", team: "Lost & Found", icon: Backpack },
  { id: "order_missing_wrong", label: "Order missing / wrong", team: "Food Operations", icon: PackageX },
  { id: "delivery_delay", label: "Delivery delay", team: "Delivery Operations", icon: Clock },
  { id: "food_quality", label: "Food quality", team: "Restaurant Quality", icon: UtensilsCrossed },
  { id: "account_promo", label: "Account & promo", team: "Account Services", icon: TicketPercent },
  { id: "app_technical", label: "App / technical", team: "Tech Support", icon: Smartphone },
  { id: "general_inquiry", label: "General inquiry", team: "Front-line Support", icon: MessageCircleQuestion },
  { id: "spam_irrelevant", label: "Spam / irrelevant", team: "Auto-close / Spam Filter", icon: Ban },
]

export const CATEGORY = Object.fromEntries(CATEGORIES.map((c) => [c.id, c])) as Record<string, CategoryInfo>

export const CHANNELS = [
  { id: "chat", label: "Chat" },
  { id: "email", label: "Email" },
  { id: "call_transcript", label: "Call transcript" },
] as const
