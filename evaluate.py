import os
import json
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

import tools

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

# --- Build a balanced evaluation sample: all real anomalies + equal number of normals ---
df = pd.read_csv("smart_meter_data.csv")

anomaly_rows = df[df["Anomaly_Label"] == "Abnormal"].sample(n=75, random_state=42).index.tolist()
normal_rows = df[df["Anomaly_Label"] == "Normal"].sample(n=75, random_state=42).index.tolist()

eval_rows = anomaly_rows + normal_rows
print(f"Evaluation sample built: {len(anomaly_rows)} anomalies + {len(normal_rows)} normals = {len(eval_rows)} total rows")

# --- Leaner tool set for fast per-row classification ---
TOOL_DEFINITIONS = [
    {
        "name": "check_weather_correlation",
        "description": "Compare this reading's consumption against similar-weather readings to judge if it's unusual.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer"}
            },
            "required": ["row_index"],
        },
    },
    {
        "name": "classify_reading",
        "description": "Give your final classification for this reading: either 'Normal' or 'Abnormal', with a brief reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {"type": "string", "enum": ["Normal", "Abnormal"]},
                "reason": {"type": "string"},
            },
            "required": ["classification", "reason"],
        },
    },
]

SYSTEM_PROMPT = """You are an AI agent classifying a single smart electricity meter reading
as Normal or Abnormal.

The vast majority of readings (about 95%) are Normal. You should classify as Abnormal ONLY
in these specific cases:
- Consumption is at or near 0 kWh (meter likely dead/disconnected)
- Consumption is more than DOUBLE the average past consumption for that meter, with no
  reasonable weather explanation
- check_weather_correlation shows the reading is more than 50% different from similar-weather
  readings

Ordinary day-to-day variation - even a 10-30% difference from average - is NORMAL. Weather-driven
changes (more usage when hot/cold) are NORMAL. If you are not highly confident it's a real
problem, classify as Normal. When in doubt, choose Normal.

You will be given the reading's details directly. You may call check_weather_correlation
once if you want more context. You MUST then call classify_reading with your final answer -
this is required, do not end your turn without calling it."""


def classify_row(row_index):
    """Runs a lean agent loop to classify a single row, returns 'Normal' or 'Abnormal'."""
    details = tools.get_reading_details(row_index)

    messages = [
        {
            "role": "user",
            "content": f"Classify this meter reading: {json.dumps(details)}",
        }
    ]

    for step in range(10):  # generous safety cap - classification is the goal, not speed
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Agent stopped without calling a tool - force it to classify
            messages.append({
                "role": "user",
                "content": "You must call classify_reading now with your final answer.",
            })
            continue

        tool_results = []
        classification_result = None

        for block in response.content:
            if block.type == "tool_use":
                if block.name == "classify_reading":
                    classification_result = block.input["classification"]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Recorded.",
                    })
                elif block.name == "check_weather_correlation":
                    result = tools.check_weather_correlation(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

        if classification_result:
            return classification_result

        if not tool_results:
            # Safety fallback: shouldn't happen, but avoid sending a broken message
            return None

        messages.append({"role": "user", "content": tool_results})

    return None


# --- Run evaluation across the sample ---
if __name__ == "__main__":
    results = []

    for i, row_index in enumerate(eval_rows):
        actual = df.iloc[row_index]["Anomaly_Label"]
        predicted = classify_row(row_index)
        results.append({"row_index": row_index, "actual": actual, "predicted": predicted})
        print(f"[{i+1}/{len(eval_rows)}] Row {row_index}: actual={actual}, predicted={predicted}")

    # --- Compute metrics ---
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)

    valid = results_df.dropna(subset=["predicted"])
    correct = (valid["actual"] == valid["predicted"]).sum()
    total = len(valid)
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    true_positives = ((valid["actual"] == "Abnormal") & (valid["predicted"] == "Abnormal")).sum()
    predicted_positives = (valid["predicted"] == "Abnormal").sum()
    precision = round((true_positives / predicted_positives) * 100, 1) if predicted_positives > 0 else 0

    actual_positives = (valid["actual"] == "Abnormal").sum()
    recall = round((true_positives / actual_positives) * 100, 1) if actual_positives > 0 else 0

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total rows evaluated: {total}")
    print(f"Overall accuracy: {accuracy}%")
    print(f"Precision (of flagged anomalies, % actually anomalous): {precision}%")
    print(f"Recall (of real anomalies, % caught): {recall}%")
    print("\nFull results saved to evaluation_results.csv")