"""Google ADK graph workflow for confirmed inbound wire-fraud calls.

The telephony adapter gathers each answer, builds ``CallInput``, then invokes this
workflow. Only the intent-classification node is a suitable place for an LLM;
authentication, routing, and CSV updates are deterministic functions.
"""

from __future__ import annotations

from google.adk import Event, Workflow
from google.adk.apps import App, ResumabilityConfig

from .tools import (
    fraud_tools,
    record_additional_wire_fraud,
    record_alerted_wire_fraud,
    record_ato_indicator,
    update_customer_contact,
    verify_customer_identity,
)
from .sub_agents import (
    ato_interview_agent,
    closing_agent,
    contact_verification_agent,
    greeting_agent,
    intent_classifier_agent,
    transaction_review_agent,
)
from pydantic import BaseModel, Field


class AuthenticationAnswers(BaseModel):
    customer_id: str
    date_of_birth: str
    last4_ssn: str = Field(min_length=4, max_length=4)


class ContactChanges(BaseModel):
    phone: str | None = None
    email: str | None = None
    address: str | None = None


class CallInput(BaseModel):
    """Answers collected across call turns by the telephony adapter."""

    authentication: AuthenticationAnswers
    call_reason: str
    alerted_transaction_id: str | None = None
    transactions_confirmed_as_fraud: list[str] = Field(default_factory=list)
    ato_answers: dict[str, bool] = Field(default_factory=dict)
    contact_changes: ContactChanges = Field(default_factory=ContactChanges)


class CallState(CallInput):
    customer_name: str | None = None
    intent: str | None = None
    authentication_passed: bool = False
    fraud_case_id: str | None = None
    related_transactions: list[dict[str, str]] = Field(default_factory=list)
    ato_flagged: bool = False
    route: str | None = None
    transcript: list[str] = Field(default_factory=list)


# The telephony controller selects these agents for individual conversation
# turns. The graph below remains the authoritative policy and write controller.
conversation_sub_agents = {
    "greeting": greeting_agent,
    "intent": intent_classifier_agent,
    "transaction_review": transaction_review_agent,
    "ato_interview": ato_interview_agent,
    "contact_verification": contact_verification_agent,
    "closing": closing_agent,
}

def greet(call: CallInput | str) -> CallState:
    # ADK Web sends the incoming turn as text; paste the structured JSON payload.
    if isinstance(call, str):
        call = CallInput.model_validate_json(call)
    state = CallState(**call.model_dump())
    state.transcript.append("Thank you for calling the Wire Fraud Response team. I will verify your identity first.")
    return state


def authenticate_customer(state: CallState) -> CallState:
    auth = state.authentication
    authentication_result = verify_customer_identity(auth.customer_id, auth.date_of_birth, auth.last4_ssn)
    state.authentication_passed = bool(authentication_result["authenticated"])
    if state.authentication_passed:
        state.customer_name = str(authentication_result["customer_name"])
        state.transcript.append("Thank you. Your identity has been verified.")
    else:
        state.transcript.append("I could not verify your identity. I will transfer you to a specialist.")
    return state


def authentication_router(state: CallState) -> Event:
    # These session values are consumed by read-only FunctionTools attached to
    # conversation sub-agents. The model never chooses the customer ID.
    session_state = {}
    if state.authentication_passed:
        session_state["authenticated_customer_id"] = state.authentication.customer_id
        if state.alerted_transaction_id:
            session_state["alerted_transaction_id"] = state.alerted_transaction_id
    return Event(
        route="AUTHENTICATED" if state.authentication_passed else "AUTH_FAILED",
        output=state,
        state=session_state,
    )


def classify_intent(state: CallState) -> CallState:
    """Safe fallback classifier. Replace with a tightly-evaluated Gemini node in production."""
    reason = state.call_reason.lower()
    wire_words = ("wire", "transfer", "beneficiary", "bank transfer")
    fraud_words = ("fraud", "scam", "unauthor", "didn't send", "did not send")
    if any(word in reason for word in wire_words) and any(word in reason for word in fraud_words):
        state.intent = "WIRE_FRAUD"
        state.transcript.append("I understand you are reporting a suspected wire transfer fraud.")
    elif any(word in reason for word in ("lost card", "mortgage", "balance", "statement")):
        state.intent = "OUT_OF_SCOPE"
        state.transcript.append("This call is outside the wire-fraud team’s scope.")
    else:
        state.intent = "NEEDS_HUMAN"
        state.transcript.append("I will connect you with a specialist to make sure we handle this correctly.")
    return state


def intent_router(state: CallState) -> Event:
    return Event(route=state.intent or "NEEDS_HUMAN", output=state)


def mark_alerted_transaction_fraud(state: CallState) -> CallState:
    if not state.alerted_transaction_id:
        raise ValueError("An alerted wire transaction ID is required for a wire-fraud report.")
    result = record_alerted_wire_fraud(state.authentication.customer_id, state.alerted_transaction_id)
    state.fraud_case_id = result["fraud_case_id"]
    state.transcript.append(f"The reported transaction has been recorded under case {state.fraud_case_id}.")
    return state


def find_recent_wires(state: CallState) -> CallState:
    state.related_transactions = fraud_tools.get_recent_wire_transactions(
        state.authentication.customer_id, state.alerted_transaction_id
    )
    if state.related_transactions:
        state.transcript.append("I found other wire transactions from the last two days for your review.")
    else:
        state.transcript.append("I found no other wire transactions from the last two days.")
    return state


def mark_confirmed_related_fraud(state: CallState) -> CallState:
    # Only transactions the caller does NOT recognize are passed in by the call UI.
    result = record_additional_wire_fraud(
        state.authentication.customer_id,
        state.transactions_confirmed_as_fraud,
        state.fraud_case_id or "",
    )
    updated = result["updated_transaction_ids"]
    if updated:
        state.transcript.append(f"I have added {len(updated)} additional transaction(s) to the fraud case.")
    return state


def assess_ato(state: CallState) -> CallState:
    # Any affirmative non-monetary account-takeover signal is escalated.
    state.ato_flagged = any(state.ato_answers.values())
    if state.ato_flagged:
        record_ato_indicator(state.authentication.customer_id)
        state.transcript.append("I have noted potential account-takeover indicators for the investigation team.")
    return state


def verify_and_update_contact(state: CallState) -> CallState:
    result = update_customer_contact(state.authentication.customer_id, **state.contact_changes.model_dump())
    changed = result["updated_fields"]
    if changed:
        state.transcript.append("Your " + ", ".join(changed) + " has been updated.")
    else:
        state.transcript.append("Your phone, email, and address have been verified.")
    return state


def route_to_wire_operations(state: CallState) -> CallState:
    state.route = "ACCOUNT_TAKEOVER_REVIEW" if state.ato_flagged else "WIRE_FRAUD_OPERATIONS"
    state.transcript.append(f"I am routing your case to {state.route.replace('_', ' ').title()} for further processing.")
    return state


def route_to_human(state: CallState) -> CallState:
    state.route = "HUMAN_FRAUD_SPECIALIST"
    state.transcript.append("I am transferring you to a fraud specialist.")
    return state


def close_call(state: CallState) -> Event:
    state.transcript.append("Thank you for calling. We will continue to work on your case. Goodbye.")
    return Event(message="\n".join(state.transcript), output=state)


# Explicit nodes and routes form the auditable call-flow graph.
root_agent = Workflow(
    name="inbound_wire_fraud_workflow",
    edges=[
        ("START", greet, authenticate_customer, authentication_router),
        (authentication_router, {"AUTHENTICATED": classify_intent, "AUTH_FAILED": route_to_human}),
        (classify_intent, intent_router),
        (intent_router, {
            "WIRE_FRAUD": mark_alerted_transaction_fraud,
            "OUT_OF_SCOPE": close_call,
            "NEEDS_HUMAN": route_to_human,
        }),
        (mark_alerted_transaction_fraud, find_recent_wires, mark_confirmed_related_fraud,
         assess_ato, verify_and_update_contact, route_to_wire_operations, close_call),
        (route_to_human, close_call),
    ],
)

# ADK application entry point used by ADK Web, API servers, and Runner instances.
app = App(
    name="inbound_wire_fraud_call",
    root_agent=root_agent,
    # Required when the telephony controller later pauses for customer answers
    # or an ADK FunctionTool confirmation and resumes the same call.
    resumability_config=ResumabilityConfig(is_resumable=True),
)
