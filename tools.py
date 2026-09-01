"""Deterministic CSV tools used by the inbound wire-fraud workflow."""

from __future__ import annotations

import csv
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools import ToolContext


class CsvFraudTools:
    """CSV-backed tools for the prototype; replace with secured DB tools in production."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir or os.getenv("FRAUD_DATA_DIR", Path(__file__).parents[1] / "data"))
        self.customers_path = self.data_dir / "customers.csv"
        self.transactions_path = self.data_dir / "transactions.csv"

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    @staticmethod
    def _write(path: Path, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def authenticate_customer(self, customer_id: str, date_of_birth: str, last4_ssn: str) -> str | None:
        """Validate the caller and return only the customer's full name."""
        for customer in self._read(self.customers_path):
            if (customer["customer_id"] == customer_id and customer["date_of_birth"] == date_of_birth
                    and customer["last4_ssn"] == last4_ssn):
                return customer["full_name"]
        return None

    def mark_transaction_fraud(self, customer_id: str, transaction_id: str) -> str:
        transactions = self._read(self.transactions_path)
        case_id = f"FRAUD-{uuid.uuid4().hex[:10].upper()}"
        matched = False
        for transaction in transactions:
            if transaction["transaction_id"] == transaction_id and transaction["customer_id"] == customer_id:
                transaction["status"] = "confirmed_fraud"
                transaction["alerted_by_customer"] = "true"
                transaction["fraud_case_id"] = case_id
                matched = True
        if not matched:
            raise ValueError("The alerted transaction does not belong to the authenticated customer.")
        self._write(self.transactions_path, transactions)
        return case_id

    def get_recent_wire_transactions(self, customer_id: str, exclude_id: str | None = None) -> list[dict[str, str]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=2)
        return [
            transaction for transaction in self._read(self.transactions_path)
            if transaction["customer_id"] == customer_id
            and transaction["channel"] == "wire"
            and transaction["transaction_id"] != exclude_id
            and datetime.fromisoformat(transaction["timestamp_utc"].replace("Z", "+00:00")) >= cutoff
        ]

    def mark_additional_fraud(self, customer_id: str, transaction_ids: list[str], case_id: str) -> list[str]:
        transactions = self._read(self.transactions_path)
        requested, updated = set(transaction_ids), []
        for transaction in transactions:
            if transaction["customer_id"] == customer_id and transaction["transaction_id"] in requested:
                transaction["status"] = "confirmed_fraud"
                transaction["fraud_case_id"] = case_id
                updated.append(transaction["transaction_id"])
        self._write(self.transactions_path, transactions)
        return updated

    def update_contact_details(self, customer_id: str, changes: dict[str, str | None]) -> list[str]:
        customers = self._read(self.customers_path)
        changed = []
        for customer in customers:
            if customer["customer_id"] == customer_id:
                for field in ("phone", "email", "address"):
                    if changes.get(field) and changes[field] != customer[field]:
                        customer[field] = changes[field]  # type: ignore[index]
                        changed.append(field)
        self._write(self.customers_path, customers)
        return changed

    def flag_account_takeover(self, customer_id: str) -> None:
        customers = self._read(self.customers_path)
        for customer in customers:
            if customer["customer_id"] == customer_id:
                customer["account_takeover_flag"] = "true"
        self._write(self.customers_path, customers)


# The graph uses this service directly. FunctionTool wrappers below are the
# narrow capabilities that may be handed to LLM sub-agents.
fraud_tools = CsvFraudTools()


def verify_customer_identity(customer_id: str, date_of_birth: str, last4_ssn: str) -> dict[str, str | bool | None]:
    """Verify the caller and return only the authentication result and full name."""
    customer_name = fraud_tools.authenticate_customer(customer_id, date_of_birth, last4_ssn)
    return {"authenticated": customer_name is not None, "customer_name": customer_name}


def lookup_recent_wire_transactions(tool_context: ToolContext) -> dict[str, object]:
    """Retrieve only the authenticated caller's recent wire transactions.

    The telephony controller sets ``authenticated_customer_id`` in session state
    immediately after successful authentication. The model cannot supply an ID.
    """
    customer_id = tool_context.state.get("authenticated_customer_id")
    alerted_transaction_id = tool_context.state.get("alerted_transaction_id")
    if not customer_id:
        return {"status": "error", "message": "Authentication is required before transaction lookup."}
    transactions = fraud_tools.get_recent_wire_transactions(customer_id, alerted_transaction_id)
    return {"transactions": transactions, "count": len(transactions)}


def record_alerted_wire_fraud(customer_id: str, transaction_id: str) -> dict[str, str]:
    """Record the caller-reported wire transaction as confirmed fraud."""
    return {"fraud_case_id": fraud_tools.mark_transaction_fraud(customer_id, transaction_id)}


def record_additional_wire_fraud(customer_id: str, transaction_ids: list[str], fraud_case_id: str) -> dict[str, object]:
    """Add customer-confirmed additional wire fraud to an existing fraud case."""
    updated = fraud_tools.mark_additional_fraud(customer_id, transaction_ids, fraud_case_id)
    return {"updated_transaction_ids": updated, "count": len(updated)}


def update_customer_contact(customer_id: str, phone: str | None = None, email: str | None = None,
                            address: str | None = None) -> dict[str, object]:
    """Update caller-confirmed customer contact details."""
    changed = fraud_tools.update_contact_details(customer_id, {"phone": phone, "email": email, "address": address})
    return {"updated_fields": changed, "count": len(changed)}


def record_ato_indicator(customer_id: str) -> dict[str, str]:
    """Set the account-takeover flag after the caller reports an ATO indicator."""
    fraud_tools.flag_account_takeover(customer_id)
    return {"status": "account_takeover_flagged"}


# Read-only tools may be attached to LLM sub-agents. Write tools require an
# explicit ADK confirmation and are intentionally not given to LLM sub-agents.
verify_customer_identity_tool = FunctionTool(verify_customer_identity)
recent_wire_transactions_tool = FunctionTool(lookup_recent_wire_transactions)
record_alerted_wire_fraud_tool = FunctionTool(record_alerted_wire_fraud, require_confirmation=True)
record_additional_wire_fraud_tool = FunctionTool(record_additional_wire_fraud, require_confirmation=True)
update_customer_contact_tool = FunctionTool(update_customer_contact, require_confirmation=True)
record_ato_indicator_tool = FunctionTool(record_ato_indicator, require_confirmation=True)
