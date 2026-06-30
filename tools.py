import pandas as pd

# Load dataset once, shared across all tool calls
df = pd.read_csv("smart_meter_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Realistic ranges for rescaling normalized 0-1 values into believable display units
# (the original dataset is min-max scaled; these ranges are just for display purposes)
RESCALE = {
    "Electricity_Consumed": (0, 10),    # kWh
    "Temperature": (-5, 40),            # Celsius
    "Humidity": (0, 100),               # %
    "Wind_Speed": (0, 30),               # km/h
    "Avg_Past_Consumption": (0, 10),    # kWh
}


def rescale(column, value):
    """Convert a 0-1 normalized value into a realistic display value."""
    low, high = RESCALE[column]
    return round(low + value * (high - low), 2)


def scan_dataset():
    """
    Returns a high-level summary of the dataset: total readings,
    and a list of row indices that look statistically unusual
    (more than 2 standard deviations from the mean consumption).
    This gives the agent a starting point before it digs into specifics.
    """
    mean = df["Electricity_Consumed"].mean()
    std = df["Electricity_Consumed"].std()
    threshold_high = mean + 2 * std
    threshold_low = mean - 2 * std

    suspicious = df[
        (df["Electricity_Consumed"] > threshold_high)
        | (df["Electricity_Consumed"] < threshold_low)
    ]

    return {
        "total_readings": len(df),
        "suspicious_row_count": len(suspicious),
        "suspicious_row_indices": suspicious.index.tolist()[:20],  # cap at 20 for readability
    }


def get_reading_details(row_index: int):
    """
    Returns full details for a single reading by its row index,
    with values rescaled to realistic units for display.
    """
    if row_index < 0 or row_index >= len(df):
        return {"error": f"Row index {row_index} is out of range (0-{len(df)-1})"}

    row = df.iloc[row_index]
    return {
        "row_index": row_index,
        "timestamp": str(row["Timestamp"]),
        "electricity_consumed_kwh": rescale("Electricity_Consumed", row["Electricity_Consumed"]),
        "temperature_celsius": rescale("Temperature", row["Temperature"]),
        "humidity_percent": rescale("Humidity", row["Humidity"]),
        "wind_speed_kmh": rescale("Wind_Speed", row["Wind_Speed"]),
        "avg_past_consumption_kwh": rescale("Avg_Past_Consumption", row["Avg_Past_Consumption"]),
    }


def check_weather_correlation(row_index: int):
    """
    Compares this reading's consumption against the average consumption
    of OTHER readings with similar temperature, to help the agent reason
    about whether a spike could be weather-driven (e.g. heating/cooling load)
    versus likely a malfunction or unusual event.
    """
    if row_index < 0 or row_index >= len(df):
        return {"error": f"Row index {row_index} is out of range (0-{len(df)-1})"}

    row = df.iloc[row_index]
    target_temp = row["Temperature"]

    # Find readings with similar temperature (within a small window)
    similar = df[(df["Temperature"] >= target_temp - 0.05) & (df["Temperature"] <= target_temp + 0.05)]
    similar_avg_consumption = similar["Electricity_Consumed"].mean()

    this_consumption = row["Electricity_Consumed"]
    difference_pct = round(((this_consumption - similar_avg_consumption) / similar_avg_consumption) * 100, 1)

    return {
        "row_index": row_index,
        "this_reading_consumption_kwh": rescale("Electricity_Consumed", this_consumption),
        "similar_weather_readings_count": len(similar),
        "similar_weather_avg_consumption_kwh": rescale("Electricity_Consumed", similar_avg_consumption),
        "difference_from_similar_weather_pct": difference_pct,
    }


def flag_anomaly(row_index: int, reason: str):
    """
    Records the agent's final decision that a reading is anomalous,
    along with its stated reason. This is what we'll later compare
    against the dataset's real Anomaly_Label to compute accuracy.
    """
    if row_index < 0 or row_index >= len(df):
        return {"error": f"Row index {row_index} is out of range (0-{len(df)-1})"}

    actual_label = df.iloc[row_index]["Anomaly_Label"]

    return {
        "row_index": row_index,
        "agent_flagged_as": "Abnormal",
        "agent_reason": reason,
        "actual_label_in_dataset": actual_label,
    }