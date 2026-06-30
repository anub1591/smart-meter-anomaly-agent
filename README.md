# smart-meter-anomaly-agent 
# ⚡ AI Pipeline Monitor Agent

An autonomous AI agent that investigates smart meter electricity data, reasons step-by-step about anomalies, and validates its own findings against labeled ground truth — built with Claude's tool-use (function-calling) API.

**[Live Demo →](https://smart-meter-agent.streamlit.app)**

---

## What this is

Most AI portfolio projects are chatbots — text in, text out. This is different: a genuine **agentic AI** system that autonomously decides which tools to call, investigates evidence across multiple steps, and reaches conclusions it can justify.

The agent:
1. Scans a 5,000-row smart meter dataset for statistical outliers
2. Pulls detailed readings on specific suspicious rows
3. Cross-references each reading against similar-weather conditions to rule out weather-driven explanations
4. Reasons out loud about whether a reading is a genuine anomaly or explainable by weather
5. Formally flags anomalies with a clear, evidence-based justification

Every step of this reasoning is visible — not just a final answer.

## Why this matters

I work as a Data Engineer on energy data for 3M+ Southern California Edison customers in my day job. This project dramatizes that real-world experience: instead of a generic demo, it's an agent solving the same kind of problem (anomaly detection in energy consumption data) that I work on professionally, but with the reasoning process fully visible.

## Validated accuracy

The agent's classifications were tested against 150 labeled readings (75 known anomalies + 75 known normal readings) from the dataset's ground-truth labels:

| Metric | Score |
|---|---|
| Precision | 74% |
| Recall | 68% |
| Overall Accuracy | 72% |

This isn't a self-reported or cherry-picked number — `evaluate.py` in this repo runs the full evaluation and is fully reproducible.

## Tech stack

- **Claude API** (Sonnet) — tool use / function calling for the agent's reasoning loop
- **Python** — agent orchestration, data tools
- **pandas** — data processing
- **Streamlit** — web interface, deployed on Streamlit Community Cloud

## How it works

The agent has four tools it can call:

- `scan_dataset()` — statistical pass over all readings to flag candidates for investigation
- `get_reading_details(row_index)` — full details on a specific reading
- `check_weather_correlation(row_index)` — compares a reading against others in similar weather, to rule out weather as an explanation
- `flag_anomaly(row_index, reason)` — formally records a conclusion with justification

Claude decides which tools to call and when, reasoning between each call, in a loop that continues until it reaches a conclusion — this is the core agentic pattern (reason → act → observe → repeat), not a fixed script.

## Project structure

```
agent.py            # Core agent orchestration loop
tools.py             # Tool functions the agent can call
evaluate.py          # Accuracy evaluation against ground-truth labels
app.py               # Streamlit web interface
requirements.txt     # Python dependencies
smart_meter_data.csv # Dataset (Kaggle: Smart Meter Electricity Consumption)
reasoning_log.txt    # Saved reasoning log powering the cached demo
evaluation_results.csv # Full row-by-row evaluation results
```

## Cost controls

Live agent runs make real API calls. The deployed app defaults to a free cached demo, with live runs rate-limited to 3 per browser session.

---

Built by [Anubhav Rastogi](https://anubhav-rastogi.com) — Data Analytics & AI Professional.
