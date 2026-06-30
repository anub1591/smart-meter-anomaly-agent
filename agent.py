import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

import tools

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

# Tool definitions Claude can see and choose to call
TOOL_DEFINITIONS = [
    {
        "name": "scan_dataset",
        "description": "Scan the full smart meter dataset and return a summary, including row indices of readings that look statistically unusual (potential anomalies to investigate further).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_reading_details",
        "description": "Get full details for a single meter reading by its row index, including consumption, temperature, humidity, and wind speed in realistic units.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer", "description": "The row index of the reading to inspect."}
            },
            "required": ["row_index"],
        },
    },
    {
        "name": "check_weather_correlation",
        "description": "Compare a specific reading's consumption against other readings recorded in similar weather conditions, to help determine if a spike could be weather-driven rather than a malfunction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer", "description": "The row index of the reading to check."}
            },
            "required": ["row_index"],
        },
    },
    {
        "name": "flag_anomaly",
        "description": "Formally flag a reading as anomalous, after reasoning through the evidence. Provide a clear, specific reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer", "description": "The row index being flagged."},
                "reason": {"type": "string", "description": "A clear explanation of why this reading is being flagged as anomalous."},
            },
            "required": ["row_index", "reason"],
        },
    },
]

# Maps tool names to the actual Python functions in tools.py
TOOL_FUNCTIONS = {
    "scan_dataset": tools.scan_dataset,
    "get_reading_details": tools.get_reading_details,
    "check_weather_correlation": tools.check_weather_correlation,
    "flag_anomaly": tools.flag_anomaly,
}

SYSTEM_PROMPT = """You are an AI agent monitoring a smart electricity meter system.
Your job is to investigate the dataset, find readings that look anomalous, and reason
step-by-step about WHY they might be anomalous (e.g. a weather-driven spike vs. a likely
meter malfunction) before formally flagging them.

Always explain your reasoning clearly and out loud before calling a tool. Investigate at
least 3 specific readings in detail using get_reading_details and check_weather_correlation
before flagging anything. When you are confident, call flag_anomaly with your conclusion.
After flagging up to 3 readings, summarize your findings and stop.
"""


def run_agent():
    messages = [
        {
            "role": "user",
            "content": "Please investigate the smart meter dataset for anomalies. Start by scanning the dataset.",
        }
    ]

    reasoning_log = []

    for step in range(15):  # safety cap on loop iterations
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Log any text reasoning Claude produced this turn
        for block in response.content:
            if block.type == "text" and block.text.strip():
                reasoning_log.append(("reasoning", block.text.strip()))
                print(f"\n[REASONING] {block.text.strip()}")

        messages.append({"role": "assistant", "content": response.content})

        # If Claude didn't call a tool, it's done
        if response.stop_reason != "tool_use":
            break

        # Handle every tool call in this turn
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                reasoning_log.append(("tool_call", f"{tool_name}({tool_input})"))
                print(f"[TOOL CALL] {tool_name}({tool_input})")

                func = TOOL_FUNCTIONS[tool_name]
                result = func(**tool_input)
                reasoning_log.append(("tool_result", result))
                print(f"[TOOL RESULT] {result}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    return reasoning_log


def run_agent_streaming():
    """
    Generator version of run_agent — yields each step as a (type, content) tuple
    so Streamlit can display reasoning live as it happens, instead of waiting
    for the full run to complete before showing anything.
    """
    messages = [
        {
            "role": "user",
            "content": "Please investigate the smart meter dataset for anomalies. Start by scanning the dataset.",
        }
    ]

    for step in range(15):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                yield ("reasoning", block.text.strip())

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                yield ("tool_call", f"{tool_name}({tool_input})")

                func = TOOL_FUNCTIONS[tool_name]
                result = func(**tool_input)
                yield ("tool_result", str(result))

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    log = run_agent()
    print("\n\n=== DONE ===")

    # Save the full reasoning log to a text file for easy review
    with open("reasoning_log.txt", "w") as f:
        for entry_type, content in log:
            if entry_type == "reasoning":
                f.write(f"[REASONING]\n{content}\n\n")
            elif entry_type == "tool_call":
                f.write(f"[TOOL CALL] {content}\n\n")
            elif entry_type == "tool_result":
                f.write(f"[TOOL RESULT] {content}\n\n")

    print("Full reasoning log saved to reasoning_log.txt")