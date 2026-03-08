"""Billing system simulator for SentinelOps Arena."""

import uuid
from typing import Dict, List

from sentinelops_arena.models import Invoice, InvoiceStatus, RefundPolicy


class BillingSystem:
    def __init__(self):
        self.invoices: Dict[str, Dict] = {}
        self.refund_policy: RefundPolicy = RefundPolicy()
        self._rate_limit: int = 0  # 0 means no limit
        self._call_count: int = 0

    def initialize(self, invoices: List[Invoice]):
        """Populate billing from Invoice models."""
        self.invoices = {inv.invoice_id: inv.model_dump() for inv in invoices}
        self.refund_policy = RefundPolicy()
        self._rate_limit = 0
        self._call_count = 0

    def check_balance(self, customer_id: str) -> Dict:
        """Return all invoices for a customer and total balance."""
        if self._rate_limit_check():
            return {"error": "Rate limit exceeded. Try again next tick."}

        customer_invoices = [
            inv for inv in self.invoices.values()
            if inv["customer_id"] == customer_id
        ]
        if not customer_invoices:
            return {"error": f"No invoices found for customer {customer_id}"}

        total = sum(
            inv["amount"] for inv in customer_invoices
            if inv["status"] in (InvoiceStatus.PENDING.value, InvoiceStatus.OVERDUE.value)
        )
        return {
            "success": True,
            "customer_id": customer_id,
            "invoices": customer_invoices,
            "outstanding_balance": total,
            "invoice_count": len(customer_invoices),
        }

    def issue_refund(self, invoice_id: str, amount: float, reason: str) -> Dict:
        """Validate refund against current policy and process it."""
        if self._rate_limit_check():
            return {"error": "Rate limit exceeded. Try again next tick."}

        if invoice_id not in self.invoices:
            return {"error": f"Invoice {invoice_id} not found"}

        invoice = self.invoices[invoice_id]

        # Check refund policy
        if amount > self.refund_policy.max_amount:
            return {
                "error": f"Refund amount ${amount:.2f} exceeds max allowed ${self.refund_policy.max_amount:.2f}"
            }

        if invoice["status"] == InvoiceStatus.REFUNDED.value:
            return {"error": f"Invoice {invoice_id} has already been refunded"}

        if amount > invoice["amount"]:
            return {
                "error": f"Refund amount ${amount:.2f} exceeds invoice amount ${invoice['amount']:.2f}"
            }

        if self.refund_policy.requires_approval:
            return {
                "success": True,
                "status": "pending_approval",
                "invoice_id": invoice_id,
                "amount": amount,
                "reason": reason,
                "message": "Refund requires manager approval under current policy",
            }

        # Process the refund
        invoice["status"] = InvoiceStatus.REFUNDED.value
        return {
            "success": True,
            "status": "refunded",
            "invoice_id": invoice_id,
            "amount": amount,
            "reason": reason,
        }

    def apply_credit(self, customer_id: str, amount: float) -> Dict:
        """Apply a credit to a customer's account by creating a credit invoice."""
        if self._rate_limit_check():
            return {"error": "Rate limit exceeded. Try again next tick."}

        credit_id = f"CREDIT-{uuid.uuid4().hex[:8].upper()}"
        credit_invoice = {
            "invoice_id": credit_id,
            "customer_id": customer_id,
            "amount": -amount,
            "status": InvoiceStatus.PAID.value,
            "date_tick": 0,
            "items": [f"Account credit: ${amount:.2f}"],
        }
        self.invoices[credit_id] = credit_invoice
        return {
            "success": True,
            "customer_id": customer_id,
            "credit_id": credit_id,
            "amount": amount,
        }

    def generate_invoice(self, customer_id: str, items: List[str], amount: float) -> Dict:
        """Create a new invoice."""
        if self._rate_limit_check():
            return {"error": "Rate limit exceeded. Try again next tick."}

        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        new_invoice = {
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "amount": amount,
            "status": InvoiceStatus.PENDING.value,
            "date_tick": 0,
            "items": items,
        }
        self.invoices[invoice_id] = new_invoice
        return {
            "success": True,
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "amount": amount,
            "items": items,
        }

    def get_current_policy(self) -> Dict:
        """Return current refund policy."""
        return {
            "success": True,
            "policy": self.refund_policy.model_dump(),
        }

    def apply_policy_drift(self, changes: Dict):
        """Modify refund policy fields."""
        data = self.refund_policy.model_dump()
        data.update(changes)
        self.refund_policy = RefundPolicy(**data)

    def set_rate_limit(self, max_calls_per_tick: int):
        """Set rate limit for API calls per tick."""
        self._rate_limit = max_calls_per_tick

    def reset_rate_limit_counter(self):
        """Reset call counter. Called each tick."""
        self._call_count = 0

    def _rate_limit_check(self) -> bool:
        """Return True if over limit."""
        self._call_count += 1
        if self._rate_limit > 0 and self._call_count > self._rate_limit:
            return True
        return False
