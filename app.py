import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="AI Pipeline Monitor Agent", layout="wide")

# --- Rate limiting: cap live agent runs per browser session ---
MAX_LIVE_RUNS_PER_SESSION = 3
if "live_run_count" not in st.session_state:
    st.session_state.live_run_count = 0

st.title("⚡ AI Pipeline Monitor Agent")
st.caption("An autonomous agent that detects anomalies in smart meter electricity data using Claude's tool-use API.")

# --- Sidebar: tech stack + accuracy metric ---
with st.sidebar:
    st.header("Tech Stack")
    st.markdown("""
    - **Claude API** (tool use / function calling)
    - **pandas** for data processing
    - **Streamlit** for this interface
    - Dataset: Kaggle Smart Meter Electricity Consumption
    """)

    st.divider()

    st.header("Validated Accuracy")
    st.metric("Precision", "74%", help="Of readings flagged as anomalous, % that were truly anomalous")
    st.metric("Recall", "68%", help="Of all real anomalies, % the agent caught")
    st.metric("Overall Accuracy", "72%", help="Tested against 150 labeled readings")
    st.caption("Validated against Kaggle dataset ground-truth labels")

# --- Load dataset for the chart ---
@st.cache_data
def load_data():
    df = pd.read_csv("smart_meter_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # Rescale 0-1 normalized consumption to realistic kWh, matching the agent's reasoning log
    df["Electricity_Consumed_kWh"] = df["Electricity_Consumed"] * 10
    return df

df = load_data()

# --- Main layout: chart on left, reasoning log on right ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📊 Consumption Over Time")

    view_option = st.radio(
        "View:", ["Daily Average (smoothed)", "Raw 30-min Readings"], horizontal=True
    )

    if view_option == "Daily Average (smoothed)":
        daily = df.set_index("Timestamp")["Electricity_Consumed_kWh"].resample("D").mean()
        st.line_chart(daily, y_label="kWh")
    else:
        chart_data = df[["Timestamp", "Electricity_Consumed_kWh"]].set_index("Timestamp")
        st.line_chart(chart_data, y_label="kWh")

    st.subheader("🔍 Run the Agent")

    remaining_runs = MAX_LIVE_RUNS_PER_SESSION - st.session_state.live_run_count
    if remaining_runs > 0:
        run_mode_options = ["View Cached Demo (free)", f"Run Live ({remaining_runs} left this session)"]
    else:
        run_mode_options = ["View Cached Demo (free)"]
        st.caption("⏸️ Live run limit reached for this session — cached demo still available.")

    demo_mode = st.radio(
        "Choose run mode:",
        run_mode_options,
        horizontal=True,
    )

    run_button = st.button("▶️ Run Agent", type="primary")

with col2:
    st.subheader("🧠 Agent Reasoning Log")
    log_container = st.container(height=500, border=True)

    if run_button:
        with log_container:
            if demo_mode == "View Cached Demo (free)":
                try:
                    with open("reasoning_log.txt", "r") as f:
                        log_content = f.read()

                    # Split into blocks by the [REASONING] / [TOOL CALL] / [TOOL RESULT] markers
                    blocks = []
                    current_label = None
                    current_text = []

                    for line in log_content.split("\n"):
                        if line.startswith("[REASONING]"):
                            if current_label:
                                blocks.append((current_label, " ".join(current_text).strip()))
                            current_label = "reasoning"
                            current_text = []
                        elif line.startswith("[TOOL CALL]"):
                            if current_label:
                                blocks.append((current_label, " ".join(current_text).strip()))
                            current_label = "tool_call"
                            current_text = [line.replace("[TOOL CALL]", "").strip()]
                        elif line.startswith("[TOOL RESULT]"):
                            if current_label:
                                blocks.append((current_label, " ".join(current_text).strip()))
                            current_label = "tool_result"
                            current_text = [line.replace("[TOOL RESULT]", "").strip()]
                        elif line.strip():
                            current_text.append(line.strip())

                    if current_label:
                        blocks.append((current_label, " ".join(current_text).strip()))

                    for label, text in blocks:
                        if not text:
                            continue
                        if label == "reasoning":
                            st.markdown(f"🧠 **Reasoning:** {text}")
                        elif label == "tool_call":
                            st.code(f"🔧 Calling tool: {text}", language=None)
                        elif label == "tool_result":
                            st.caption(f"📊 Result: {text}")
                        time.sleep(0.3)  # simulate live streaming feel

                except FileNotFoundError:
                    st.error("reasoning_log.txt not found - run agent.py first to generate it.")
            elif demo_mode.startswith("Run Live"):
                if st.session_state.live_run_count >= MAX_LIVE_RUNS_PER_SESSION:
                    st.warning("Live run limit reached for this session. Try the cached demo instead!")
                else:
                    st.session_state.live_run_count += 1
                    st.info("Running live agent — this may take 2-3 minutes and uses API credits...")
                    try:
                        from agent import run_agent_streaming
                        for label, text in run_agent_streaming():
                            # Clean up numpy's float64 repr for nicer display
                            text = text.replace("np.float64(", "").replace(")", "")
                            if label == "reasoning":
                                st.markdown(f"🧠 **Reasoning:** {text}")
                            elif label == "tool_call":
                                st.code(f"🔧 Calling tool: {text}", language=None)
                            elif label == "tool_result":
                                st.caption(f"📊 Result: {text}")
                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")
    else:
        with log_container:
            st.info("Click 'Run Agent' to see the reasoning process here.")

st.divider()
st.caption("Built by Anubhav Rastogi — demonstrating agentic AI tool-use, not just generative chat.")