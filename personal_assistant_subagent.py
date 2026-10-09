from datetime import date
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 1. Define Low-Level API Tools (Stubs)

@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,    # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


# 2. Model Initialization and Specialized Sub-Agents

model = init_chat_model("gpt-5.5")

CALENDAR_AGENT_PROMPT = (
    f"Today's date is {date.today().isoformat()}. "
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "If there is no suitable time slot, stop and confirm unavailability in your response. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        ),
    ],
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        ),
    ],
)


# 3. Wrap Sub-Agents as Tools for the Supervisor Agent

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language. Use this when the user wants to create, modify, or check calendar appointments. Handles date/time parsing, availability checking, and event creation. Input: Natural language scheduling request (e.g., 'meeting with design team next Tuesday at 2pm')"""
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language. Use this when the user wants to send notifications, reminders, or any email communication. Handles recipient extraction, subject generation, and email composition. Input: Natural language email request (e.g., 'send them a reminder about the meeting')"""
    result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


# 4. Supervisor Agent Setup

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence or in parallel as appropriate."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)


# 5. Execution Example (Demonstrating Multi-Domain Request & Human-in-the-Loop)

if __name__ == "__main__":
    query = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )
    config = {"configurable": {"thread_id": "personal-assistant-demo"}}
    interrupts = []

    print("--- Stream Phase 1: Execution up to Human Interruption ---")
    stream = supervisor_agent.stream_events(
        {"messages": [{"role": "user", "content": query}]},
        config,
        version="v3",
    )

    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name} ({item.input})")

        if stream.interrupted:
            for interrupt_ in stream.interrupts:
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")

    # Inspect interrupts and provide decisions
    print("\n--- Processing Human Approvals / Edits ---")
    resume = {}
    for interrupt_ in interrupts:
        for request in interrupt_.value["action_requests"]:
            print(f"Pending Decision: {request['description']}")

            # Example: Approve calendar event, Edit email subject line
            if request["action"]["tool_name"] == "send_email":
                edited_action = request.copy()
                edited_action["args"]["subject"] = "Mockups reminder"
                resume[interrupt_.id] = {
                    "decisions": [{"type": "edit", "edited_action": edited_action}]
                }
            else:
                resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    print("\n--- Stream Phase 2: Resume After Approval ---")
    stream = supervisor_agent.stream_events(
        Command(resume=resume),
        config,
        version="v3",
    )

    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name} ({item.input})")