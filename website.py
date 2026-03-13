# Capstone Project – Live Heart Rate + Hydration Monitoring Dashboard (Streamlit)

# This app simulates a wearable monitoring dashboard using a 2-hour, 1 Hz dataset.

# Run it with:
#   streamlit run website.py

import time  # controls playback timing for the streaming loop
from pathlib import Path  # safe file paths that work on Windows/Mac/Linux
from datetime import datetime, timedelta  # live timestamp stream + time offsets

import numpy as np  # used for NaN handling + interpolation support
import pandas as pd  # dataframe operations + Excel loading
import streamlit as st  # Streamlit UI framework
import plotly.express as px  # interactive charts


# ============================================================
# Continuous Monitoring Logic
# ============================================================

class ContinuousHealthMonitor:

    def __init__(
        self,
        hr_threshold_low=30,                 # user min HR
        hr_threshold_high=220,               # user max HR
        hydration_threshold_low=0.30,        # user min hydration (0–1)
        hydration_threshold_high=1,       # user max hydration (0–1)
        hr_hold_ticks=5,                     # how many ticks HR must persist abnormal 5s
        hyd_hold_ticks=2,                    # how many ticks hydration must persist abnormal 10min
        hr_disconnect_hold_ticks=5,          # how many ticks HR missing before persistent disconnect
        hyd_disconnect_hold_ticks=5,         # how many ticks hyd missing before persistent disconnect
    ):
        # Store thresholds as floats so comparisons are consistent
        self.hr_threshold_low = float(hr_threshold_low)
        self.hr_threshold_high = float(hr_threshold_high)
        self.hydration_threshold_low = float(hydration_threshold_low)
        self.hydration_threshold_high = float(hydration_threshold_high)

        # Hold times in ticks (UI ticks, not necessarily 1 second if you use update_every > 1)
        self.hr_hold_ticks = int(hr_hold_ticks)
        self.hyd_hold_ticks = int(hyd_hold_ticks)
        self.hr_disconnect_hold_ticks = int(hr_disconnect_hold_ticks)
        self.hyd_disconnect_hold_ticks = int(hyd_disconnect_hold_ticks)

        # Persistence counters (how long the issue has lasted)
        self.abnormal_hr_count = 0
        self.abnormal_hydration_count = 0
        self.hr_missing_count = 0
        self.hyd_missing_count = 0

        # Persistent warning states
        self.hr_disconnection_warning = False
        self.hydration_disconnection_warning = False
        self.hr_abnormal_warning = False
        self.hydration_abnormal_warning = False

        # Simulated actuator state (vibration motor idea)
        self.motor_on = False

        # Alarm history for UI display (list of dicts)
        self.alarm_history = []  # each row: time,type,source,message

    def _log_alarm(self, live_dt: datetime, alarm_type, message, is_clear=False):
        # Write alarms using the "live stream timestamp" so it matches the simulated time.
        timestamp = live_dt.strftime("%Y-%m-%d %H:%M:%S")

        self.alarm_history.append(
            {
                "time": timestamp,
                "type": "CLEAR" if is_clear else "ALARM",
                "source": alarm_type,
                "message": message,
            }
        )

        # Limit the list length so the app stays responsive
        if len(self.alarm_history) > 80:
            self.alarm_history.pop(0)

    def is_hr_abnormal(self, hr_value):
        # HR abnormal means outside user min/max (but NaN is handled by disconnect logic) (new change ~87)
        if hr_value is None or pd.isna(hr_value):
            return False
        hr_value = float(hr_value)
        return (hr_value < self.hr_threshold_low) or (hr_value > self.hr_threshold_high)
    
    def is_hydration_abnormal(self, hydration_value):
        # Hydration abnormal means outside user min/max (0–1 scale)
        if hydration_value is None or pd.isna(hydration_value):
            return False
        hydration_value = float(hydration_value)
        return (hydration_value < self.hydration_threshold_low) or (hydration_value > self.hydration_threshold_high)

    def process_hr(self, live_dt: datetime, hr_value):
        # Run on every UI tick with the current HR value (raw value)
        # This updates persistent states + alarm history.

        # Missing HR counts as a disconnection
        if hr_value is None or pd.isna(hr_value):
            self.hr_missing_count += 1

            # Only trigger persistent disconnect after hold ticks
            if self.hr_missing_count >= self.hr_disconnect_hold_ticks and not self.hr_disconnection_warning:
                self.hr_disconnection_warning = True
                self._log_alarm(live_dt, "HR Disconnect", "Heart rate signal missing")

            self.update_motor_state(live_dt)
            return

        # HR is present -> reset missing counter
        self.hr_missing_count = 0

        # If we were disconnected and now got a sample, clear the disconnect warning
        if self.hr_disconnection_warning:
            self.hr_disconnection_warning = False
            self._log_alarm(live_dt, "HR Disconnect", "Reconnected", is_clear=True)

        # Abnormal HR persistence logic (outside min/max for hr_hold_ticks)
        if self.is_hr_abnormal(hr_value):
            self.abnormal_hr_count += 1
            if self.abnormal_hr_count >= self.hr_hold_ticks and not self.hr_abnormal_warning:
                self.hr_abnormal_warning = True
                self._log_alarm(
                    live_dt,
                    "HR Abnormal",
                    f"{float(hr_value):.1f} BPM for {self.abnormal_hr_count} ticks",
                )
        else:
            # If HR returns to normal, clear persistent alarm (if it was on)
            if self.hr_abnormal_warning:
                self.hr_abnormal_warning = False
                self._log_alarm(live_dt, "HR Abnormal", "Returned to normal", is_clear=True)
            self.abnormal_hr_count = 0

        # Update motor after updating warning states
        self.update_motor_state(live_dt)

    def process_hydration(self, live_dt: datetime, hydration_value):
        # Run on every UI tick with current hydration value (0–1)
        # Same persistence logic concept as HR.

        if hydration_value is None or pd.isna(hydration_value):
            self.hyd_missing_count += 1

            if self.hyd_missing_count >= self.hyd_disconnect_hold_ticks and not self.hydration_disconnection_warning:
                self.hydration_disconnection_warning = True
                self._log_alarm(live_dt, "Hydration Disconnect", "Hydration signal missing")

            self.update_motor_state(live_dt)
            return

        # Hydration present -> reset missing counter
        self.hyd_missing_count = 0

        # Clear persistent disconnect if it was active
        if self.hydration_disconnection_warning:
            self.hydration_disconnection_warning = False
            self._log_alarm(live_dt, "Hydration Disconnect", "Reconnected", is_clear=True)

        # Abnormal hydration persistence logic
        if self.is_hydration_abnormal(hydration_value):
            self.abnormal_hydration_count += 1
            if self.abnormal_hydration_count >= self.hyd_hold_ticks and not self.hydration_abnormal_warning:
                self.hydration_abnormal_warning = True
                self._log_alarm(
                    live_dt,
                    "Hydration Abnormal",
                    f"{float(hydration_value):.3f} for {self.abnormal_hydration_count} ticks",
                )
        else:
            if self.hydration_abnormal_warning:
                self.hydration_abnormal_warning = False
                self._log_alarm(live_dt, "Hydration Abnormal", "Returned to normal", is_clear=True)
            self.abnormal_hydration_count = 0

        self.update_motor_state(live_dt)

    def update_motor_state(self, live_dt: datetime):
        # Motor turns ON if any persistent alarm is active
        new_motor_state = (
            self.hr_disconnection_warning
            or self.hydration_disconnection_warning
            or self.hr_abnormal_warning
            or self.hydration_abnormal_warning
        )

        # Only log when the motor state changes
        if new_motor_state != self.motor_on:
            self.motor_on = new_motor_state
            self._log_alarm(live_dt, "Motor", f"Motor turned {'ON' if self.motor_on else 'OFF'}")

    def get_active_warnings(self):
        # Returns the persistent alarms currently active (not the instant flags)
        warnings = []
        if self.hr_disconnection_warning:
            warnings.append("🔴 HR sensor disconnected / missing (persistent)")
        if self.hydration_disconnection_warning:
            warnings.append("🔴 Hydration sensor disconnected / missing (persistent)")
        if self.hr_abnormal_warning:
            warnings.append("🔴 HR outside user range persisted")
        if self.hydration_abnormal_warning:
            warnings.append("🔴 Hydration outside user range persisted")
        return warnings

    def get_status(self):
        # Returns current states in a dict (useful for UI quick checks)
        return {
            "motor_on": self.motor_on,
            "hr_disconnect": self.hr_disconnection_warning,
            "hyd_disconnect": self.hydration_disconnection_warning,
            "hr_abnormal": self.hr_abnormal_warning,
            "hyd_abnormal": self.hydration_abnormal_warning,
        }


# ============================================================
# Config (paths + dataset)
# ============================================================

# Find folder that contains this main.py file
BASE_DIR = Path(__file__).resolve().parent

# Your sensor-only Excel file (must be in same folder as main.py unless you change this)
EXCEL_FILE = BASE_DIR / "hr_hydration_training_2h_SENSOR_ONLY.xlsx"

# Sheet name
SHEET = "Sheet1"

# Streamlit page settings
st.set_page_config(page_title="Capstone Live Monitor", page_icon="💧", layout="wide")


# ============================================================
# Minimal CSS styling 
# ============================================================

# Light styling to make it look like a dashboard (no functional impact)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 14px 14px;
        border-radius: 16px;
      }
      .pill {
        display:inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        font-size: 0.85rem;
      }
      .muted {opacity:0.75}
      hr {border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 0.8rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading + baseline auto flags + clean HR
# ============================================================

@st.cache_data
def load_data():
    # Load the Excel file and create columns used by the UI
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    # Read the sheet
    raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET)

    # Validate expected columns exist
    if "heart_rate_bpm" not in raw.columns or "hydration_0to1" not in raw.columns:
        raise ValueError("Expected columns: heart_rate_bpm and hydration_0to1")

    # Standardize into the names used by the dashboard
    df = pd.DataFrame()
    df["hr_bpm_raw"] = pd.to_numeric(raw["heart_rate_bpm"], errors="coerce")
    df["hydration_ui_0to1"] = pd.to_numeric(raw["hydration_0to1"], errors="coerce")

    # Baseline flags (NOT user threshold flags).
    # These are "general sensor" markers used mainly for clean signal building + markers.
    df["hr_flag_disconnect"] = df["hr_bpm_raw"].isna().astype(int)

    # Baseline outlier/artifact rules (hard-coded)
    hr = df["hr_bpm_raw"]
    jump = (hr - hr.shift(1)).abs()
    range_outlier = (hr < 30) | (hr > 220)
    jump_outlier = jump > 40
    df["hr_flag_outlier_or_artifact"] = (range_outlier | jump_outlier).fillna(False).astype(int)

    # Clean HR signal for display:
    # - remove baseline outliers
    # - interpolate across missing
    # - lightly smooth (3-sample moving average)
    hr_clean = hr.copy()
    hr_clean[df["hr_flag_outlier_or_artifact"] == 1] = np.nan
    hr_clean = hr_clean.interpolate(limit_direction="both")
    hr_clean = hr_clean.rolling(3, min_periods=1).mean()
    df["hr_bpm_clean"] = hr_clean

    # Placeholder column (not used directly in this version, but kept for compatibility)
    df["hydr_flag_possible_disorder"] = 0

    return df


# Load data and fail nicely if something is wrong
try:
    df = load_data()
except Exception as e:
    st.error("Could not load dataset.")
    st.write(f"Expected file at: `{EXCEL_FILE}`")
    st.exception(e)
    st.stop()


# ============================================================
# Sidebar controls (user thresholds + display)
# ============================================================

st.sidebar.title("⚙️ Settings")

# These profile inputs are just for realism (not used for computations right now)
st.sidebar.markdown("### User Profile Inputs")
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=25, key="age")
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, key="weight")
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, key="height")
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"], key="gender")

# Calculate personalized thresholds based on profile (new change ~396)
@st.cache_data
def calculate_personalized_thresholds(age, weight, height, gender):
    """Calculate personalized HR and hydration thresholds"""
    
    # Maximum heart rate (traditional formula)
    hr_max = 220 - age
    
    # Resting heart rate varies by gender and fitness
    if gender == "Female":
        resting_hr = 65  # Females slightly higher resting HR
    elif gender == "Male":
        resting_hr = 60
    else:
        resting_hr = 62
    
    # Heart rate zones (percentage of max)
    hr_normal_min = resting_hr  # Lower bound of normal
    hr_normal_max = int(0.85 * hr_max)  # 85% of max is upper normal
    
    # Hydration thresholds based on body weight
    # Total Body Water (TBW) estimate: ~60% of body weight for men, ~50% for women
    if gender == "Female":
        tbw_percent = 0.50
    elif gender == "Male":
        tbw_percent = 0.60
    else:
        tbw_percent = 0.55
    
    total_body_water_liters = weight * tbw_percent
    
    # Dehydration threshold: loss of 2% of body weight from water
    dehydration_threshold = 1.0 - (0.02 * weight / total_body_water_liters)
    
    # Overhydration threshold (rare, but for safety)
    overhydration_threshold = 1.05  # 5% above normal
    
    return {
        'hr_max': hr_max,
        'resting_hr': resting_hr,
        'hr_normal_min': hr_normal_min,
        'hr_normal_max': hr_normal_max,
        'dehydration_threshold': max(0.3, dehydration_threshold),  # Don't go below 0.3
        'overhydration_threshold': overhydration_threshold,
        'total_body_water_liters': total_body_water_liters
    }

# Calculate personalized thresholds
personalized = calculate_personalized_thresholds(age, weight, height, gender)

# Display personalized info in sidebar
with st.sidebar.expander("Your Personalized Metrics", expanded=False):
    st.write(f"**Max HR:** {personalized['hr_max']} bpm")
    st.write(f"**Resting HR:** {personalized['resting_hr']} bpm")
    st.write(f"**Normal HR Range:** {personalized['hr_normal_min']}-{personalized['hr_normal_max']} bpm")
    st.write(f"**Total Body Water:** {personalized['total_body_water_liters']:.1f} L")
    st.write(f"**Dehydration threshold:** {personalized['dehydration_threshold']:.2f}")


# # Preset modes only set default values in the sidebar
# mode = st.sidebar.selectbox("Profile", ["General", "Athlete", "Sleep", "Disorder-safe (flag only)"], index=0)

# # Preset defaults
# if mode == "General":
#     default_hr_min, default_hr_max, default_jump = 45, 185, 40
#     default_hyd_min, default_hyd_max = 0.30, 1.00
#     auto_clean_default = True
# if mode == "Athlete":
#     default_hr_min, default_hr_max, default_jump = 50, 205, 50
#     default_hyd_min, default_hyd_max = 0.20, 1.00
#     auto_clean_default = True
# elif mode == "Sleep":
#     default_hr_min, default_hr_max, default_jump = 30, 100, 30
#     default_hyd_min, default_hyd_max = 0.30, 1.00
#     auto_clean_default = True
# else:
#     default_hr_min, default_hr_max, default_jump = 45, 220, 60
#     default_hyd_min, default_hyd_max = 0.10, 1.00
#     auto_clean_default = False

# ============================================================
# Thresholds - NOW USING PERSONALIZED VALUES AS DEFAULTS
# ============================================================
st.sidebar.markdown("### Alert Thresholds") #new change ~482

# Option to use personalized defaults or manual
use_personalized = st.sidebar.checkbox("Use personalized thresholds", value=True)

default_jump = 40
auto_clean_default = True

if use_personalized:
    # Use calculated personalized values
    default_hr_min = personalized['hr_normal_min']
    default_hr_max = personalized['hr_normal_max']
    default_hyd_min = personalized['dehydration_threshold']
    default_hyd_max = personalized['overhydration_threshold']
    
    # Show that personalized mode is active
    st.sidebar.success("✅ Using your personalized thresholds")
else:
    # Manual mode with presets
    mode = st.sidebar.selectbox("Profile Preset", ["General", "Athlete", "Sleep", "Disorder-safe"], index=0)
    
    if mode == "General":
        default_hr_min, default_hr_max = 45, 185
        default_hyd_min, default_hyd_max = 0.30, 1.00
    elif mode == "Athlete":
        default_hr_min, default_hr_max = 40, 200
        default_hyd_min, default_hyd_max = 0.20, 1.00
    elif mode == "Sleep":
        default_hr_min, default_hr_max = 40, 100
        default_hyd_min, default_hyd_max = 0.30, 1.00
    else:  # Disorder-safe
        default_hr_min, default_hyd_min = 30, 0.10
        default_hr_max, default_hyd_max = 220, 1.00
    
    default_jump = 40  # Default jump threshold
    auto_clean_default = True

with st.sidebar.expander("Set Thresholds Manually", expanded=not use_personalized):
    hr_min = st.number_input("Heart Rate min (bpm)", 
                            value=float(default_hr_min), 
                            step=1.0,
                            help="Below this triggers low HR alert")
    
    hr_max = st.number_input("Heart Rate max (bpm)", 
                            value=float(default_hr_max), 
                            step=1.0,
                            help="Above this triggers high HR alert")

    hyd_min = st.number_input("Hydration min (0–1)", 
                             value=float(default_hyd_min), 
                             step=0.05, 
                             format="%.2f",
                             help="Below this indicates dehydration")
    
    hyd_max = st.number_input("Hydration max (0–1)", 
                             value=float(default_hyd_max), 
                             step=0.05, 
                             format="%.2f",
                             help="Above this may indicate overhydration")

    max_delta = st.number_input("Max HR jump (bpm/s)", 
                               value=40.0, 
                               step=1.0,
                               help="Sudden jumps above this are flagged as artifacts")

# User-defined thresholds (this is what you wanted: user chooses min/max)
with st.sidebar.expander("Thresholds", expanded=True):
    hr_min = st.number_input("Heart Rate min (bpm)", value=float(default_hr_min), step=1.0)
    hr_max = st.number_input("Heart Rate max (bpm)", value=float(default_hr_max), step=1.0)

    hyd_min = st.number_input("Hydration min (0–1)", value=float(default_hyd_min), step=0.05, format="%.2f")
    hyd_max = st.number_input("Hydration max (0–1)", value=float(default_hyd_max), step=0.05, format="%.2f")

    # Max HR jump threshold used for artifact detection (bpm per second)
    max_delta = st.number_input("Max HR jump (bpm/s)", value=float(default_jump), step=1.0)

    # Auto-clean affects display only (clean vs raw)
    auto_clean = st.checkbox(
        "Auto-clean outliers ",
        value=auto_clean_default,
        help="If ON, the dashboard displays hr_bpm_clean. If OFF, it displays hr_bpm_raw."
    )

# Display / streaming controls
with st.sidebar.expander("Display", expanded=False):
    # Speed is playback multiplier (10 means 10x faster than real time)
    speed = st.slider("Demo speed (x)", 1, 30, 1)

    # Update interval controls how many rows you advance each tick
    update_every = st.selectbox(
        "Update interval (seconds)",
        [1, 2, 5],
        index=0,
        help="Shows the data every N seconds. Dataset is 1 Hz, so this jumps by N rows per tick."
    )

    # Window length for plotting (how much history to show)
    window_s = st.slider("Chart window (seconds)", 60, 900, 60, step=30)

    # Smoothing affects plot appearance only (not logged values)
    smoothing = st.selectbox("Smoothing", ["None", "Moving average (5s)", "Moving average (15s)"], index=0)

    # Optional overlay + marker toggles
    show_flag_markers = st.checkbox("Show outlier/disconnect markers", value=True)

# Sidebar dataset info
st.sidebar.markdown("---")
st.sidebar.caption("Data")
st.sidebar.write("File:", f"`{EXCEL_FILE.name}`")
st.sidebar.write("Sheet:", f"`{SHEET}`")
st.sidebar.write("Rows:", f"`{len(df):,}`")


# ============================================================
# Session state (stream index + running + live timestamp + logs)
# ============================================================

# Current dataset index (stream position)
if "i" not in st.session_state:
    st.session_state.i = 0

# Whether playback is running
if "running" not in st.session_state:
    st.session_state.running = False

# Anchor for live timestamps (so the demo time looks realistic)
if "stream_start_dt" not in st.session_state:
    st.session_state.stream_start_dt = datetime.now().replace(microsecond=0)

# Per-sample log (saves flags per tick)
if "flag_log" not in st.session_state:
    st.session_state.flag_log = []

# Transition log (flag ON/OFF changes)
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# Last flags saved (to detect transitions)
if "last_flags" not in st.session_state:
    st.session_state.last_flags = None

# Previous HR and timestamp for jump-based artifact detection
if "prev_hr_for_flags" not in st.session_state:
    st.session_state.prev_hr_for_flags = None

if "prev_dt_for_flags" not in st.session_state:
    st.session_state.prev_dt_for_flags = None

# Persistent monitor instance (stateful alarms)
if "monitor" not in st.session_state:
    st.session_state.monitor = ContinuousHealthMonitor(
        hr_threshold_low=hr_min,
        hr_threshold_high=hr_max,
        hydration_threshold_low=hyd_min,
        hydration_threshold_high=hyd_max,
    )

# Keep monitor thresholds aligned with sidebar every rerun
st.session_state.monitor.hr_threshold_low = float(hr_min)
st.session_state.monitor.hr_threshold_high = float(hr_max)
st.session_state.monitor.hydration_threshold_low = float(hyd_min)
st.session_state.monitor.hydration_threshold_high = float(hyd_max)

# Clamp i so we never index out of bounds
st.session_state.i = max(0, min(int(st.session_state.i), max(len(df) - 1, 0)))


# ============================================================
# Helper functions
# ============================================================
def get_battery_percent_placeholder():
    # Placeholder for now (until ESP32 BLE battery characteristic is wired in)
    return None  # change later to an int 0–100

def apply_smoothing(series: pd.Series) -> pd.Series:
    # Optional smoothing for plots only (does not change logged values)
    if smoothing == "Moving average (5s)":
        return series.rolling(5, min_periods=1).mean()
    if smoothing == "Moving average (15s)":
        return series.rolling(15, min_periods=1).mean()
    return series

def choose_hr(row):
    # Choose which HR value the dashboard displays
    return row["hr_bpm_clean"] if auto_clean else row["hr_bpm_raw"]

def compute_dynamic_flags(live_dt: datetime, row, step_seconds: int):

    hr_raw = row["hr_bpm_raw"]
    hyd = row["hydration_ui_0to1"]

    # Missing flags
    hr_missing = int(pd.isna(hr_raw))
    hyd_missing = int(pd.isna(hyd))

    # User min/max range flags
    hr_out_of_range = 0
    if not pd.isna(hr_raw):
        hr_out_of_range = int((float(hr_raw) < float(hr_min)) or (float(hr_raw) > float(hr_max)))

    hyd_out_of_range = 0
    if not pd.isna(hyd):
        hyd_out_of_range = int((float(hyd) < float(hyd_min)) or (float(hyd) > float(hyd_max)))

    # Artifact/jump detection (uses previous HR and previous time)
    hr_artifact = 0
    if (
        not pd.isna(hr_raw)
        and st.session_state.prev_hr_for_flags is not None
        and st.session_state.prev_dt_for_flags is not None
    ):
        # dt is in seconds between ticks (based on live timestamps, not dataset timestamps)
        dt = max((live_dt - st.session_state.prev_dt_for_flags).total_seconds(), 1e-6)

        # allowed jump = (bpm/s) * seconds
        allowed_jump = float(max_delta) * dt

        # if jump is bigger than allowed, flag as artifact
        if abs(float(hr_raw) - float(st.session_state.prev_hr_for_flags)) > allowed_jump:
            hr_artifact = 1

    # Update previous HR/time for next tick (only when HR is present)
    if not pd.isna(hr_raw):
        st.session_state.prev_hr_for_flags = float(hr_raw)
        st.session_state.prev_dt_for_flags = live_dt

    return {
        "hr_missing": hr_missing,
        "hyd_missing": hyd_missing,
        "hr_out_of_range": hr_out_of_range,
        "hyd_out_of_range": hyd_out_of_range,
        "hr_artifact": hr_artifact,
    }

def append_flag_log(live_dt: datetime, idx: int, row, flags: dict):
    # Save one row into the per-sample flag log (timestamp + values + flags)
    st.session_state.flag_log.append(
        {
            "timestamp_live": live_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sample": int(idx),
            "hr_bpm_raw": None if pd.isna(row["hr_bpm_raw"]) else float(row["hr_bpm_raw"]),
            "hydration_0to1": None if pd.isna(row["hydration_ui_0to1"]) else float(row["hydration_ui_0to1"]),
            **flags,
        }
    )

    # Keep memory reasonable
    if len(st.session_state.flag_log) > 5000:
        st.session_state.flag_log = st.session_state.flag_log[-5000:]

def log_flag_transitions(live_dt: datetime, flags: dict):
    # Save an ON/OFF event whenever any flag changes state.
    ts = live_dt.strftime("%Y-%m-%d %H:%M:%S")
    last = st.session_state.last_flags

    # On the first tick, log ON states only
    if last is None:
        for k, v in flags.items():
            if int(v) == 1:
                st.session_state.event_log.append({"timestamp_live": ts, "flag": k, "state": "ON"})
        st.session_state.last_flags = flags.copy()
        return

    # Compare current vs last and record transitions
    for k, v in flags.items():
        if int(v) != int(last.get(k, 0)):
            st.session_state.event_log.append(
                {"timestamp_live": ts, "flag": k, "state": "ON" if int(v) == 1 else "OFF"}
            )

    # Update last flags
    st.session_state.last_flags = flags.copy()

    # Trim to last N transitions
    if len(st.session_state.event_log) > 400:
        st.session_state.event_log = st.session_state.event_log[-400:]


# ============================================================
# Header (title + running badge)
# ============================================================

title_col, badge_col = st.columns([3, 1])

with title_col:
    # Main page title
    st.markdown("## ❤️💧 Capstone Heart Rate & Hydration Monitor")

with badge_col:
    # Small status pill at top right
    status = "RUNNING ✅" if st.session_state.running else "STOPPED ⏸️"
    st.markdown(f'<div class="pill">{status}</div>', unsafe_allow_html=True)


# ============================================================
# Controls row (Start / Stop / Reset + progress)
# ============================================================

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:
    if st.button("▶ Start", use_container_width=True):
        # Re-anchor timestamps so the current sample aligns with "now"
        st.session_state.stream_start_dt = datetime.now().replace(microsecond=0) - timedelta(seconds=int(st.session_state.i))
        st.session_state.running = True

with c2:
    if st.button("⏸ Stop", use_container_width=True):
        st.session_state.running = False

with c3:
    if st.button("↩ Reset", use_container_width=True):
        # Reset stream index and stop playback
        st.session_state.i = 0
        st.session_state.running = False

        # Reset live timestamp base
        st.session_state.stream_start_dt = datetime.now().replace(microsecond=0)

        # Clear logs
        st.session_state.flag_log = []
        st.session_state.event_log = []
        st.session_state.last_flags = None

        # Clear artifact detection history
        st.session_state.prev_hr_for_flags = None
        st.session_state.prev_dt_for_flags = None

        # Reset persistent monitor state
        st.session_state.monitor = ContinuousHealthMonitor(
            hr_threshold_low=hr_min,
            hr_threshold_high=hr_max,
            hydration_threshold_low=hyd_min,
            hydration_threshold_high=hyd_max,
        )

        # Rerun so UI updates instantly
        st.rerun()

with c4:
    # Progress bar through the dataset
    prog = (st.session_state.i + 1) / max(len(df), 1)
    st.progress(prog)
    st.caption(f"Stream position: {st.session_state.i + 1:,}/{len(df):,}  ({prog*100:.1f}%)")

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs(["📟 Dashboard", "📈 Trends", "🧾 Data"])


# ============================================================
# Current row + live timestamp (based on anchor + sample index)
# ============================================================

# Current stream index and step size (how many rows we advance per tick)
i = int(st.session_state.i)
step = int(update_every)

# Current row data
row = df.iloc[i]

# HR displayed (clean or raw depending on checkbox)
hr_used = choose_hr(row)

# Hydration value (always raw)
hyd_used = row["hydration_ui_0to1"]

# Compute the live timestamp for this sample
live_dt = st.session_state.stream_start_dt + timedelta(seconds=i)

# Compute dynamic flags (these are the saved flags the user wanted)
dyn_flags = compute_dynamic_flags(live_dt, row, step_seconds=step)

# Append logs (per-sample log + transitions)
append_flag_log(live_dt, i, row, dyn_flags)
log_flag_transitions(live_dt, dyn_flags)

# Feed persistent monitor using live timestamps
st.session_state.monitor.process_hr(live_dt, row["hr_bpm_raw"])
st.session_state.monitor.process_hydration(live_dt, hyd_used)

# Persistent warnings (hold-based) + monitor states
warnings = st.session_state.monitor.get_active_warnings()
monitor_status = st.session_state.monitor.get_status()


# ============================================================
# Tab 1: Dashboard (metrics + alarms + logs)
# ============================================================

with tab1:
    # Four top metrics cards
    k1, k2, k3, k4= st.columns(4)

    with k1:
        st.metric("HR (bpm)", "—" if pd.isna(hr_used) else f"{float(hr_used):.1f}")

    with k2:
        st.metric("Hydration (0–1)", "—" if pd.isna(hyd_used) else f"{float(hyd_used):.3f}")

    with k3:
        st.metric("Live Timestamp", live_dt.strftime("%Y-%m-%d %H:%M:%S"))

    battery_pct = get_battery_percent_placeholder()

    with k4:
        st.metric(
            "Battery (%)",
            "—" if battery_pct is None else f"{int(battery_pct)}%"
        )
    st.markdown("<hr/>", unsafe_allow_html=True)

    # LEFT: alarms + alarm history (more important)
    # RIGHT: flags + flag event log + quick checks
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Persistent alerts (hold-based) are the "real alarms"
        if warnings:
            st.warning("### ⏳ Persistent Alerts (monitor)\n" + "\n".join([f"- {w}" for w in warnings]))
        else:
            st.success("### ✅ No persistent alarms\nNo conditions have persisted long enough to trigger an alarm.")

        # Alarm history is shown directly (not hidden)
        st.markdown("### 📋 Alarm History ")
        if st.session_state.monitor.alarm_history:
            history_df = pd.DataFrame(st.session_state.monitor.alarm_history)
            st.dataframe(history_df.iloc[::-1].head(25), use_container_width=True, hide_index=True)
        else:
            st.caption("No alarms have been triggered yet.")

        # new change ~879
        st.markdown("### 💡 Personalized Health Tips")
        # Generate tips based on current readings and profile
        tips = []

        # HR-based tips
        if not pd.isna(hr_used):
            hr_current = float(hr_used)
            if hr_current > personalized['hr_normal_max'] * 0.9:
                tips.append("• Your heart rate is approaching your max. Consider resting.")
            elif hr_current < personalized['resting_hr'] * 0.9 and hr_current > 40:
                tips.append("• Your heart rate is lower than usual for your profile. This could indicate good fitness or need for check.")
        # Hydration tips based on gender and weight
        if not pd.isna(hyd_used):
            hyd_current = float(hyd_used)
            daily_water_needs = weight * 0.033  # 33ml per kg
            if hyd_current < personalized['dehydration_threshold'] * 1.1:
                tips.append(f"• You may need hydration. Based on your weight ({weight}kg), aim for {daily_water_needs:.1f}L daily.")
        
        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success("• All readings normal for your profile!")

    with right_col:
        # Active flags this tick (instant flags)
        active_now = []

        if dyn_flags["hr_out_of_range"]:
            active_now.append("🔴 HR outside user range")
        if dyn_flags["hyd_out_of_range"]:
            active_now.append("🔴 Hydration outside user range")
        if dyn_flags["hr_artifact"]:
            active_now.append("🔴 HR artifact (sudden jump)")
        if dyn_flags["hr_missing"]:
            active_now.append("🔴 HR missing (disconnect)")
        if dyn_flags["hyd_missing"]:
            active_now.append("🔴 Hydration missing (disconnect)")

        if active_now:
            st.error("### ⚠️ Active Flags \n" + "\n".join([f"- {x}" for x in active_now]))
        else:
            st.success("### ✅ No active flags ")

        st.markdown("#### Quick checks")
        hr_ok = (not pd.isna(row["hr_bpm_raw"])) and (float(hr_min) <= float(row["hr_bpm_raw"]) <= float(hr_max))
        hyd_ok = (not pd.isna(hyd_used)) and (float(hyd_min) <= float(hyd_used) <= float(hyd_max))

        st.write(f"- HR in user range: **{'Yes' if hr_ok else 'No'}**")
        st.write(f"- Hydration in user range: **{'Yes' if hyd_ok else 'No'}**")
        st.write(f"- Auto-clean display: **{'On' if auto_clean else 'Off'}**")
        st.write(f"- Update interval: **{step}s**")

        motor_state = "⚡ ON" if monitor_status["motor_on"] else "⏹️ OFF"
        st.write(f"- Motor state: **{motor_state}**")

        # Flag event log moved here (less important than alarms)
        st.markdown("### 🧾 Flag Event Log ")
        log_df = pd.DataFrame(st.session_state.flag_log)
        flag_cols = ["hr_out_of_range", "hyd_out_of_range", "hr_artifact", "hr_missing", "hyd_missing"]

        events_df = log_df[log_df[flag_cols].any(axis=1)].copy()
        events_df = events_df.iloc[::-1]  # newest first
        st.dataframe(events_df.head(25), use_container_width=True, hide_index=True)

        # Keep transitions in an expander (optional)
        with st.expander("🔁 Flag transitions (Last 12)"):
            ev = pd.DataFrame(st.session_state.event_log)
            if len(ev) > 0:
                st.dataframe(ev.iloc[::-1].head(12), use_container_width=True, hide_index=True)
            else:
                st.caption("No transitions yet.")

    # Profile summary expander (new change ~918)
    with st.expander("👤 Your Profile Summary", expanded=False):
        prof_col1, prof_col2, prof_col3 = st.columns(3)
        with prof_col1:
            st.metric("Age", f"{age} years")
            st.metric("Gender", gender)
        with prof_col2:
            st.metric("Weight", f"{weight} kg")
            st.metric("Height", f"{height} cm")
        with prof_col3:
            st.metric("BMI", f"{weight/((height/100)**2):.1f}")
            st.metric("Est. Body Water", f"{personalized['total_body_water_liters']:.1f} L")
        
        st.caption("Your thresholds are personalized based on these values")

# ============================================================
# Tab 2: Trends (charts)
# ============================================================

with tab2:
    # Determine plot window bounds
    WINDOW = int(window_s)
    lo = max(0, i - WINDOW)

    # Slice dataset for recent window
    df_window = df.iloc[lo:i + 1].copy()

    # Create a live timestamp series for the window (aligned to stream_start_dt)
    base_dt = st.session_state.stream_start_dt + timedelta(seconds=lo)
    df_window["timestamp_live"] = [base_dt + timedelta(seconds=int(k)) for k in range(len(df_window))]

    # Build HR signals for plotting
    df_window["hr_used"] = df_window["hr_bpm_clean"] if auto_clean else df_window["hr_bpm_raw"]
    df_window["hr_used_sm"] = apply_smoothing(df_window["hr_used"])
    df_window["hr_raw_sm"] = apply_smoothing(df_window["hr_bpm_raw"])

    # Build hydration smoothed column for plotting (avoids px.line label weirdness)
    df_window["hyd_sm"] = apply_smoothing(df_window["hydration_ui_0to1"])

    # Pull matching saved flags for markers (so chart markers match what’s logged)
    log_df = pd.DataFrame(st.session_state.flag_log)
    log_window = log_df[(log_df["sample"] >= lo) & (log_df["sample"] <= i)].copy()

    # Convert timestamps back into datetime for plotting markers
    if len(log_window) > 0:
        log_window["timestamp_live"] = pd.to_datetime(log_window["timestamp_live"])

    left, right = st.columns(2)

    with left:
        # HR chart (smoothed)
        # Single-line HR plot only (no overlay)
        hr_fig = px.line(
            df_window,
            x="timestamp_live",
            y="hr_used_sm",
            title="Heart Rate (recent)",
            labels={"hr_used_sm": "HR (bpm)", "timestamp_live": "Time"},
        )

        # Marker points from saved flag log (not baseline flags)
        if show_flag_markers and len(log_window) > 0:
            pts = log_window[
                (log_window["hr_out_of_range"] == 1)
                | (log_window["hr_artifact"] == 1)
                | (log_window["hr_missing"] == 1)
            ].copy()

            if len(pts) > 0:
                s = px.scatter(
                    pts,
                    x="timestamp_live",
                    y="hr_bpm_raw",
                    hover_data=["hr_out_of_range", "hr_artifact", "hr_missing"],
                ).data[0]
                s.name = "Flagged HR points (saved)"
                hr_fig.add_trace(s)

        hr_fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hr_fig, use_container_width=True)

    with right:
        # Hydration chart (smoothed)
        hy_fig = px.line(
            df_window,
            x="timestamp_live",
            y="hyd_sm",
            title="Hydration (recent)",
            labels={"timestamp_live": "Time", "hyd_sm": "Hydration (0–1)"},
        )

        # Hydration markers from saved log
        if show_flag_markers and len(log_window) > 0:
            pts2 = log_window[(log_window["hyd_out_of_range"] == 1) | (log_window["hyd_missing"] == 1)].copy()

            if len(pts2) > 0:
                s2 = px.scatter(
                    pts2,
                    x="timestamp_live",
                    y="hydration_0to1",
                    hover_data=["hyd_out_of_range", "hyd_missing"],
                ).data[0]
                s2.name = "Flagged hydration points (saved)"
                hy_fig.add_trace(s2)

        hy_fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hy_fig, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("Markers come from your saved flag log, so the chart matches what is being recorded.")


# ============================================================
# Tab 3: Data (preview + downloads)
# ============================================================

with tab3:
    # Show raw dataset rows near the current index
    st.subheader("Data preview (last 25 rows)")
    st.dataframe(df.iloc[max(0, i - 25): i + 1], use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Show session logs (saved during this run)
    st.subheader("Saved logs (session)")
    log_df = pd.DataFrame(st.session_state.flag_log)
    ev_df = pd.DataFrame(st.session_state.event_log)

    st.markdown("**Per-sample flag log (latest 50):**")
    st.dataframe(log_df.iloc[::-1].head(50), use_container_width=True, hide_index=True)

    st.markdown("**Flag transition log (latest 50):**")
    if len(ev_df) > 0:
        st.dataframe(ev_df.iloc[::-1].head(50), use_container_width=True, hide_index=True)
    else:
        st.caption("No transitions yet.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Download buttons for logs
    cdl1, cdl2 = st.columns(2)

    with cdl1:
        # Download just the current window of the log (matches the chart window concept)
        window_log = log_df[(log_df["sample"] >= max(0, i - int(window_s))) & (log_df["sample"] <= i)].copy()

        st.download_button(
            "⬇️ Download current window flag log (CSV)",
            data=window_log.to_csv(index=False).encode("utf-8"),
            file_name="capstone_window_flag_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with cdl2:
        # Download entire session log
        st.download_button(
            "⬇️ Download full flag log (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="capstone_full_flag_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# Stream loop (advance by 1/2/5 seconds per tick)
# ============================================================

# When running is true, move forward and rerun after sleeping
if st.session_state.running:
    if st.session_state.i >= len(df) - 1:
        # Stop at the end of the dataset
        st.session_state.running = False
        st.toast("Reached end of dataset.", icon="✅")
    else:
        # Advance by step rows (step is 1/2/5 seconds worth of samples)
        st.session_state.i = min(st.session_state.i + step, len(df) - 1)

        # Sleep scaled by speed
        time.sleep(step / max(speed, 1))

        # Force rerun to show next "frame"
        st.rerun()


# ============================================================
# Footer
# ============================================================

# Small footer text
st.markdown(
    '<div class="muted" style="margin-top: 0.6rem;">Built for capstone demo • Streamlit</div>',
    unsafe_allow_html=True,
)
