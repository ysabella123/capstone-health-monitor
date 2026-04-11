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

import os  # for file path operations
import base64  # for encoding data (if needed for downloads or BLE transmission)
import json  # for saving logs or configs (if implemented)

CRYPTO_AVAILABLE = True
try:
    from cryptography.fernet import Fernet  # for data encryption (if implemented)
    from cryptography.hazmat.primitives import hashes  # for key hashing if needed
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # for key derivation if needed
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # for encryption if needed
    from cryptography.hazmat.backends import default_backend  # for encryption backend
except Exception as e:
    CRYPTO_AVAILABLE = False
    Fernet = None
    hashes = None
    PBKDF2HMAC = None
    Cipher = None
    algorithms = None
    modes = None
    default_backend = None
    print(f"Cryptography import not available: {e}")

import asyncio
import threading
import queue

BLE_AVAILABLE = True
try:
    from bleak import BleakScanner, BleakClient
except Exception as e:
    BLE_AVAILABLE = False
    BleakScanner = None
    BleakClient = None
    print(f"BLE import not available: {e}")

import sys
import os
from pathlib import Path

# ============================================================
# BLE / receive1 setup
# ============================================================

# Get the current script's directory
current_dir = Path(__file__).resolve().parent
print(f"📁 Script directory: {current_dir}")

receive1 = None
receive1_file = current_dir / "receive1.py"
print(f"🔧 Looking for receive1.py at: {receive1_file}")

if BLE_AVAILABLE and receive1_file.exists():
    try:
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
            print(f"✅ Added {current_dir} to Python path")
        import receive1
        print(f"✅ Successfully imported receive1 from: {receive1.__file__}")
        print(f"Available in receive1: {[x for x in dir(receive1) if not x.startswith('_')]}")
    except Exception as e:
        print(f"⚠️ Import error: {e}")
        receive1 = None
else:
    if not BLE_AVAILABLE:
        print("⚠️ BLE not available on this deployment")
    if not receive1_file.exists():
        print(f"⚠️ receive1.py not found at {receive1_file}")


# ============================================================
# Continuous Monitoring Logic
# ============================================================

class ContinuousHealthMonitor:

    def __init__(
        self,
        hr_threshold_low=30,                 # user min HR
        hr_threshold_high=220,               # user max HR
        hydration_threshold_low=-5.0,        # user min hydration change (%)
        hydration_threshold_high=5.0,         # user max hydration change (%)
        spo2_threshold_low=95.0,             # user min blood oxygen (%)
        spo2_threshold_high=100.0,           # user max blood oxygen (%)
        hr_hold_ticks=5,                     # how many ticks HR must persist abnormal 5s
        hyd_hold_ticks=2,                    # how many ticks hydration must persist abnormal 10min
        spo2_hold_ticks=3,                   # how many ticks SpO₂ must persist abnormal
        hr_disconnect_hold_ticks=5,          # how many ticks HR missing before persistent disconnect
        hyd_disconnect_hold_ticks=5,         # how many ticks hyd missing before persistent disconnect
        spo2_disconnect_hold_ticks=5,        # how many ticks SpO₂ missing before persistent disconnect
    ):
        # Store thresholds as floats so comparisons are consistent
        self.hr_threshold_low = float(hr_threshold_low)
        self.hr_threshold_high = float(hr_threshold_high)
        self.hydration_threshold_low = float(hydration_threshold_low)
        self.hydration_threshold_high = float(hydration_threshold_high)
        self.spo2_threshold_low = float(spo2_threshold_low)
        self.spo2_threshold_high = float(spo2_threshold_high)

        # Hold times in ticks (UI ticks, not necessarily 1 second if you use update_every > 1)
        self.hr_hold_ticks = int(hr_hold_ticks)
        self.hyd_hold_ticks = int(hyd_hold_ticks)
        self.spo2_hold_ticks = int(spo2_hold_ticks)
        self.hr_disconnect_hold_ticks = int(hr_disconnect_hold_ticks)
        self.hyd_disconnect_hold_ticks = int(hyd_disconnect_hold_ticks)
        self.spo2_disconnect_hold_ticks = int(spo2_disconnect_hold_ticks)

        # Persistence counters (how long the issue has lasted)
        self.abnormal_hr_count = 0
        self.abnormal_hydration_count = 0
        self.abnormal_spo2_count = 0
        self.hr_missing_count = 0
        self.hyd_missing_count = 0
        self.spo2_missing_count = 0

        # Persistent warning states
        self.hr_disconnection_warning = False
        self.hydration_disconnection_warning = False
        self.spo2_disconnection_warning = False
        self.hr_abnormal_warning = False
        self.hydration_abnormal_warning = False
        self.spo2_abnormal_warning = False

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
        # Hydration abnormal means outside user min/max (% change scale)
        if hydration_value is None or pd.isna(hydration_value):
            return False
        hydration_value = float(hydration_value)
        return (hydration_value < self.hydration_threshold_low) or (hydration_value > self.hydration_threshold_high)

    def is_spo2_abnormal(self, spo2_value):
        # Blood oxygen abnormal means outside user min/max (% scale)
        if spo2_value is None or pd.isna(spo2_value):
            return False
        spo2_value = float(spo2_value)
        return (spo2_value < self.spo2_threshold_low) or (spo2_value > self.spo2_threshold_high)

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
        # Run on every UI tick with current hydration value (% change)
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
                    f"{float(hydration_value):.1f}% for {self.abnormal_hydration_count} ticks",
                )
        else:
            if self.hydration_abnormal_warning:
                self.hydration_abnormal_warning = False
                self._log_alarm(live_dt, "Hydration Abnormal", "Returned to normal", is_clear=True)
            self.abnormal_hydration_count = 0

        self.update_motor_state(live_dt)

    def process_spo2(self, live_dt: datetime, spo2_value):
        # Run on every UI tick with current blood oxygen value (%)
        # Same persistence logic concept as HR and hydration.

        if spo2_value is None or pd.isna(spo2_value):
            self.spo2_missing_count += 1

            if self.spo2_missing_count >= self.spo2_disconnect_hold_ticks and not self.spo2_disconnection_warning:
                self.spo2_disconnection_warning = True
                self._log_alarm(live_dt, "SpO₂ Disconnect", "Blood oxygen signal missing")

            self.update_motor_state(live_dt)
            return

        # SpO₂ present -> reset missing counter
        self.spo2_missing_count = 0

        # Clear persistent disconnect if it was active
        if self.spo2_disconnection_warning:
            self.spo2_disconnection_warning = False
            self._log_alarm(live_dt, "SpO₂ Disconnect", "Reconnected", is_clear=True)

        # Abnormal SpO₂ persistence logic
        if self.is_spo2_abnormal(spo2_value):
            self.abnormal_spo2_count += 1
            if self.abnormal_spo2_count >= self.spo2_hold_ticks and not self.spo2_abnormal_warning:
                self.spo2_abnormal_warning = True
                self._log_alarm(
                    live_dt,
                    "SpO₂ Abnormal",
                    f"{float(spo2_value):.1f}% for {self.abnormal_spo2_count} ticks",
                )
        else:
            if self.spo2_abnormal_warning:
                self.spo2_abnormal_warning = False
                self._log_alarm(live_dt, "SpO₂ Abnormal", "Returned to normal", is_clear=True)
            self.abnormal_spo2_count = 0

        self.update_motor_state(live_dt)

    def update_motor_state(self, live_dt: datetime):
        # Motor turns ON if any persistent alarm is active
        new_motor_state = (
            self.hr_disconnection_warning
            or self.hydration_disconnection_warning
            or self.spo2_disconnection_warning
            or self.hr_abnormal_warning
            or self.hydration_abnormal_warning
            or self.spo2_abnormal_warning
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
        if self.spo2_disconnection_warning:
            warnings.append("🔴 Blood oxygen sensor disconnected / missing (persistent)")
        if self.hr_abnormal_warning:
            warnings.append("🔴 HR outside user range persisted")
        if self.hydration_abnormal_warning:
            warnings.append("🔴 Hydration outside user range persisted")
        if self.spo2_abnormal_warning:
            warnings.append("🔴 Blood oxygen outside user range persisted")
        return warnings

    def get_status(self):
        # Returns current states in a dict (useful for UI quick checks)
        return {
            "motor_on": self.motor_on,
            "hr_disconnect": self.hr_disconnection_warning,
            "hyd_disconnect": self.hydration_disconnection_warning,
            "spo2_disconnect": self.spo2_disconnection_warning,
            "hr_abnormal": self.hr_abnormal_warning,
            "hyd_abnormal": self.hydration_abnormal_warning,
            "spo2_abnormal": self.spo2_abnormal_warning,
        }


class HealthDataEncryptor:
    def __init__(self, key_file=None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography package is not installed on this deployment")
        self.backend = default_backend()
        self.key = self._load_or_create_key(key_file)
        self.cipher_suite = Fernet(self.key)

    def _load_or_create_key(self, key_file):
        if key_file and Path(key_file).exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            if key_file:
                Path(key_file).parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                st.sidebar.success(f"🔑 New encryption key saved to {key_file}")
            return key
    
    def encrypt_data(self, data):
        # Convert to JSON string if dict/DataFrame
        if isinstance(data, dict):
            data_str = json.dumps(data, default=str)
        elif hasattr(data, 'to_json'):  # pandas DataFrame
            data_str = data.to_json()
        else:
            data_str = str(data)
        
        # Encrypt
        encrypted = self.cipher_suite.encrypt(data_str.encode())
        return encrypted
    
    def decrypt_data(self, encrypted_data):
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            st.error(f"Decryption failed: {e}")
            return None
        
    def encrypt_file(self, input_path, output_path=None):
        if not output_path:
            output_path = str(input_path) + '.encrypted'
        
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        encrypted = self.cipher_suite.encrypt(file_data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        return output_path
    
    def decrypt_file(self, input_path, output_path=None):
        if not output_path:
            output_path = str(input_path).replace('.encrypted', '.decrypted')
        
        with open(input_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.cipher_suite.decrypt(encrypted)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        return output_path
    
class PasswordBasedEncryptor:
    def __init__(self, password, salt=None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography package is not installed on this deployment")
        self.salt = salt or os.urandom(16)
        self.key = self._derive_key(password, self.salt)

    def _derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_user_data(self, data):
        cipher = Fernet(self.key)
        return cipher.encrypt(json.dumps(data, default=str).encode())
    
    def decrypt_user_data(self, encrypted_data):
        cipher = Fernet(self.key)
        decrypted = cipher.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

class SecureDataTransmitter:
    def __init__(self, encryptor):
        self.encryptor = encryptor

    def prepare_for_transmission(self, data, include_metadata=False):
        packet = {
            'encrypted_data': self.encryptor.encrypt_data(data),
            'timestamp': datetime.now().isoformat()
        }
        
        if include_metadata:
            # Add integrity hash
            packet['checksum'] = self._calculate_checksum(packet['encrypted_data'])
        
        return packet
    
    def _calculate_checksum(self, data):
        import hashlib
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def verify_packet(packet):
        if 'checksum' in packet:
            import hashlib
            calculated = hashlib.sha256(packet['encrypted_data']).hexdigest()
            return calculated == packet['checksum']
        return True

# ============================================================
# BLE Receiver Integration
# ============================================================

# Create a queue for passing data from BLE thread to Streamlit
ble_data_queue = queue.Queue()

# Global BLE state
ble_state = {
    "connected": False,
    "heart_rate": None,
    "hydration": None,
    "spo2": None,
    "battery": None,
    "uptime": None,
    "last_update": None,
    "device_name": "",
    "device_addr": "",
    "running": False,
    "error": None
}

# def ble_worker():
#     """Run the BLE receiver in a separate thread"""
#     try:
#         # Use asyncio to run the main BLE function
#         asyncio.run(run_ble_receiver())
#     except Exception as e:
#         ble_state["error"] = str(e)
#         ble_state["running"] = False

# async def run_ble_receiver():
#     """Modified version of receive1's main function that updates our state"""
#     global ble_state
    
#     while ble_state["running"]:
#         device = await receive1.find_device()
#         if device is None:
#             # No device found, wait and retry
#             await asyncio.sleep(2)
#             continue
        
#         try:
#             async with BleakClient(device, timeout=15.0) as client:
#                 ble_state["connected"] = True
#                 ble_state["device_name"] = device.name or device.address
#                 ble_state["device_addr"] = device.address
#                 ble_state["last_update"] = time.time()
                
#                 # Subscribe to characteristics
#                 subs = [
#                     (receive1.CHARACTERISTIC_UUID, parse_combined, "Combined"),
#                     (receive1.HEART_RATE_CHAR_UUID, parse_heart_rate_streamlit, "Heart Rate"),
#                     (receive1.HYDRATION_CHAR_UUID, parse_hydration_streamlit, "Hydration"),
#                     (receive1.BATTERY_CHAR_UUID, parse_battery_streamlit, "Battery"),
#                 ]
                
#                 for uuid, handler, label in subs:
#                     try:
#                         await client.start_notify(uuid, handler)
#                     except Exception as e:
#                         print(f"Could not subscribe to {label}: {e}")
                
#                 # Keep connection alive
#                 while ble_state["running"] and client.is_connected:
#                     await asyncio.sleep(1)
                    
#                 ble_state["connected"] = False
                
#         except Exception as e:
#             ble_state["error"] = str(e)
#             ble_state["connected"] = False
#             await asyncio.sleep(3)

def parse_combined_streamlit(sender, data: bytearray):
    """Parse combined data and update Streamlit state"""
    try:
        text = data.decode("utf-8").strip()
        ble_state["last_update"] = time.time()
        
        for part in text.split(","):
            part = part.strip()
            if part.startswith("Uptime:"):
                ble_state["uptime"] = part[7:]
            elif part.startswith("HR:"):
                ble_state["heart_rate"] = float(part[3:])
            elif part.startswith("Hyd:"):
                ble_state["hydration"] = float(part[4:].rstrip("%"))
            elif part.startswith("SpO2:"):
                ble_state["spo2"] = float(part[5:].rstrip("%"))
            elif part.startswith("SpO₂:"):
                ble_state["spo2"] = float(part[5:].rstrip("%"))
        
        # Also put in queue for Streamlit to process
        ble_data_queue.put({
            "type": "combined",
            "data": ble_state.copy()
        })
        
    except Exception as e:
        print(f"Parse error: {e}")

def parse_heart_rate_streamlit(sender, data: bytearray):
    """Parse heart rate data"""
    try:
        import struct
        if len(data) == 2:
            heart_rate = float(data[1])
        elif len(data) == 4:
            heart_rate = struct.unpack("<f", data)[0]
        else:
            return
            
        if 0 < heart_rate < 300:
            ble_state["heart_rate"] = heart_rate
            ble_state["last_update"] = time.time()
            ble_data_queue.put({
                "type": "heart_rate",
                "value": heart_rate,
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"HR parse error: {e}")

def parse_hydration_streamlit(sender, data: bytearray):
    """Parse hydration data"""
    try:
        hydration = float(data.decode("utf-8").strip().rstrip("%"))
        ble_state["hydration"] = hydration
        ble_state["last_update"] = time.time()
        ble_data_queue.put({
            "type": "hydration",
            "value": hydration,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Hydration parse error: {e}")

def parse_spo2_streamlit(sender, data: bytearray):
    """Parse blood oxygen data"""
    try:
        spo2 = float(data.decode("utf-8").strip().rstrip("%"))
        ble_state["spo2"] = spo2
        ble_state["last_update"] = time.time()
        ble_data_queue.put({
            "type": "spo2",
            "value": spo2,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"SpO₂ parse error: {e}")

def parse_battery_streamlit(sender, data: bytearray):
    """Parse battery data"""
    try:
        if len(data) >= 1:
            battery = int(data[0])
            ble_state["battery"] = battery
            ble_state["last_update"] = time.time()
            ble_data_queue.put({
                "type": "battery",
                "value": battery,
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"Battery parse error: {e}")

async def run_ble_receiver():
    """Run the BLE receiver using receive1 functions"""
    global ble_state

    if not BLE_AVAILABLE or receive1 is None:
        ble_state["error"] = "BLE/receive1 not available on this deployment"
        return
    
    print("Starting BLE receiver...")
    
    while ble_state["running"]:
        try:
            # Use receive1's find_device function
            device = await receive1.find_device()
            
            if device is None:
                print("Device not found, waiting...")
                await asyncio.sleep(2)
                continue
            
            print(f"Found device: {device.name or device.address}")
            
            async with BleakClient(device, timeout=15.0) as client:
                ble_state["connected"] = True
                ble_state["device_name"] = device.name or device.address
                ble_state["device_addr"] = device.address
                ble_state["last_update"] = time.time()
                ble_state["error"] = None
                
                print(f"Connected to {ble_state['device_name']}")
                
                # Use constants from receive1
                # Check if the constants exist, if not, use defaults
                try:
                    combined_uuid = getattr(receive1, 'CHARACTERISTIC_UUID', "0000abcd-0000-1000-8000-00805f9b34fb")
                    hr_uuid = getattr(receive1, 'HEART_RATE_CHAR_UUID', "00002a37-0000-1000-8000-00805f9b34fb")
                    hyd_uuid = getattr(receive1, 'HYDRATION_CHAR_UUID', "abcdef04-1234-5678-9abc-def012345678")
                    spo2_uuid = getattr(receive1, 'SPO2_CHAR_UUID', None)
                    batt_uuid = getattr(receive1, 'BATTERY_CHAR_UUID', "00002a19-0000-1000-8000-00805f9b34fb")
                except:
                    combined_uuid = "0000abcd-0000-1000-8000-00805f9b34fb"
                    hr_uuid = "00002a37-0000-1000-8000-00805f9b34fb"
                    hyd_uuid = "abcdef04-1234-5678-9abc-def012345678"
                    spo2_uuid = None
                    batt_uuid = "00002a19-0000-1000-8000-00805f9b34fb"
                
                subs = [
                    (combined_uuid, parse_combined_streamlit, "Combined"),
                    (hr_uuid, parse_heart_rate_streamlit, "Heart Rate"),
                    (hyd_uuid, parse_hydration_streamlit, "Hydration"),
                    (batt_uuid, parse_battery_streamlit, "Battery"),
                ]

                if spo2_uuid:
                    subs.append((spo2_uuid, parse_spo2_streamlit, "SpO₂"))
                
                for uuid, handler, label in subs:
                    try:
                        await client.start_notify(uuid, handler)
                        print(f"Subscribed to {label}")
                    except Exception as e:
                        print(f"Could not subscribe to {label}: {e}")
                
                # Keep connection alive
                while ble_state["running"] and client.is_connected:
                    await asyncio.sleep(1)
                    
                ble_state["connected"] = False
                print("Disconnected")
                
        except Exception as e:
            ble_state["error"] = str(e)
            ble_state["connected"] = False
            print(f"Connection error: {e}")
            await asyncio.sleep(3)

def ble_worker():
    """Run the BLE receiver in a separate thread"""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(run_ble_receiver())
        loop.close()
    except Exception as e:
        ble_state["error"] = str(e)
        ble_state["running"] = False
        print(f"BLE worker error: {e}")

def start_ble_receiver():
    """Start the BLE receiver thread"""
    if not BLE_AVAILABLE or receive1 is None:
        ble_state["error"] = "BLE/receive1 not available on this deployment"
        return False

    if not ble_state["running"]:
        ble_state["running"] = True
        ble_state["error"] = None
        ble_thread = threading.Thread(target=ble_worker, daemon=True)
        ble_thread.start()
        return True
    return False

def stop_ble_receiver():
    """Stop the BLE receiver"""
    ble_state["running"] = False
    ble_state["connected"] = False

def send_command_to_device(command):
    """Send a command to the ESP32 via BLE"""
    if ble_state["connected"]:
        ble_data_queue.put({
            "type": "command",
            "command": command
        })
        return True
    return False

def process_ble_data():
    """Process queued BLE data and update the health monitor"""
    try:
        while not ble_data_queue.empty():
            data = ble_data_queue.get_nowait()
            
            if data["type"] == "heart_rate":
                # Update monitor with real HR data
                current_time = datetime.now()
                if "monitor" in st.session_state:
                    st.session_state.monitor.process_hr(current_time, data["value"])
                
            elif data["type"] == "hydration":
                # Update monitor with real hydration data
                current_time = datetime.now()
                # Convert percentage to 0-1 scale if needed
                hydration_value = data["value"]
                if "monitor" in st.session_state:
                    st.session_state.monitor.process_hydration(current_time, hydration_value)
                
    except queue.Empty:
        pass

def add_ble_controls_to_sidebar():
    """Add BLE connection controls to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.title("📡 BLE Connection")

    if not BLE_AVAILABLE or receive1 is None:
        st.sidebar.warning("BLE is not available on this deployment")
    
    # BLE control buttons
    col_ble1, col_ble2 = st.sidebar.columns(2)
    with col_ble1:
        if st.button("🔌 Connect BLE", use_container_width=True):
            if start_ble_receiver():
                st.sidebar.success("BLE receiver started!")
            else:
                st.sidebar.warning("BLE receiver already running")
    
    with col_ble2:
        if st.button("⏸ Disconnect BLE", use_container_width=True):
            stop_ble_receiver()
            st.sidebar.info("BLE receiver stopped")
    
    # Display BLE status
    if ble_state["connected"]:
        st.sidebar.success(f"✅ Connected to: {ble_state['device_name']}")
        st.sidebar.write(f"📱 Address: {ble_state['device_addr']}")
        if ble_state["last_update"]:
            ago = time.time() - ble_state["last_update"]
            if ago < 5:
                st.sidebar.info(f"📊 Last update: {ago:.1f}s ago")
            else:
                st.sidebar.warning(f"⚠️ Last update: {ago:.1f}s ago")
    else:
        if ble_state["running"]:
            st.sidebar.info("🔍 Searching for device...")
        else:
            st.sidebar.info("❌ Not connected")
        
        if ble_state["error"]:
            st.sidebar.error(f"Error: {ble_state['error']}")
    
    # BLE Command controls
    if ble_state["connected"]:
        st.sidebar.markdown("### Device Commands")
        col_cmd1, col_cmd2, col_cmd3 = st.sidebar.columns(3)
        
        with col_cmd1:
            if st.button("🔔 Vibrate ON", use_container_width=True):
                if send_command_to_device("MOTOR_ON"):
                    st.sidebar.success("Command sent")
                else:
                    st.sidebar.warning("Not connected")
        
        with col_cmd2:
            if st.button("🔕 Vibrate OFF", use_container_width=True):
                if send_command_to_device("MOTOR_OFF"):
                    st.sidebar.success("Command sent")
                else:
                    st.sidebar.warning("Not connected")
        
        with col_cmd3:
            if st.button("🔄 Reset", use_container_width=True):
                if send_command_to_device("RESET"):
                    st.sidebar.success("Command sent")
                else:
                    st.sidebar.warning("Not connected")

# ============================================================
# Config (paths + dataset)
# ============================================================

# Find folder that contains this main.py file
BASE_DIR = Path(__file__).resolve().parent

# Your sensor-only Excel file (must be in same folder as main.py unless you change this)
EXCEL_FILE = BASE_DIR / "hr_hydration_training_2h_SENSOR_ONLY_with_spo2.xlsx"

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
    df["hydration_percent"] = pd.to_numeric(raw["hydration_0to1"], errors="coerce") * 100.0

    # Blood oxygen column support
    # This accepts a few possible Excel column names and standardizes them to spo2_percent
    if "blood_oxygen_percent" in raw.columns:
        df["spo2_percent"] = pd.to_numeric(raw["blood_oxygen_percent"], errors="coerce")
    elif "spo2_percent" in raw.columns:
        df["spo2_percent"] = pd.to_numeric(raw["spo2_percent"], errors="coerce")
    elif "oxygen_percent" in raw.columns:
        df["spo2_percent"] = pd.to_numeric(raw["oxygen_percent"], errors="coerce")
    else:
        # If no blood oxygen data exists yet, create an empty column so the UI still works
        df["spo2_percent"] = np.nan

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
    dehydration_threshold = -5.0
    
    # Overhydration threshold (rare, but for safety)
    overhydration_threshold = 5.0  # 5% above normal
    
    return {
        'hr_max': hr_max,
        'resting_hr': resting_hr,
        'hr_normal_min': hr_normal_min,
        'hr_normal_max': hr_normal_max,
        'dehydration_threshold': dehydration_threshold,
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
    st.write(f"**Hydration change range:** {personalized['dehydration_threshold']:.1f}% to {personalized['overhydration_threshold']:.1f}%")


# # Preset modes only set default values in the sidebar
# mode = st.sidebar.selectbox("Profile", ["General", "Athlete", "Sleep", "Disorder-safe (flag only)"], index=0)

# # Preset defaults
# if mode == "General":
#     default_hr_min, default_hr_max, default_jump = 45, 185, 40
#     default_hyd_min, default_hyd_max = -5.0, 5.0
#     auto_clean_default = True
# if mode == "Athlete":
#     default_hr_min, default_hr_max, default_jump = 50, 205, 50
#     default_hyd_min, default_hyd_max = -5.0, 5.0
#     auto_clean_default = True
# elif mode == "Sleep":
#     default_hr_min, default_hr_max, default_jump = 30, 100, 30
#     default_hyd_min, default_hyd_max = -5.0, 5.0
#     auto_clean_default = True
# else:
#     default_hr_min, default_hr_max, default_jump = 45, 220, 60
#     default_hyd_min, default_hyd_max = -5.0, 5.0
#     auto_clean_default = False
default_jump = 40
auto_clean_default = True
# ============================================================
# Thresholds - NOW USING PERSONALIZED VALUES AS DEFAULTS
# ============================================================
st.sidebar.markdown("### Alert Thresholds") #new change ~482

# Option to use personalized defaults or manual
use_personalized = st.sidebar.checkbox("Use personalized thresholds", value=True, 
                                       help="Base thresholds on your age/gender/weight")

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
        default_hyd_min, default_hyd_max = -5.0, 5.0
    elif mode == "Athlete":
        default_hr_min, default_hr_max = 40, 200
        default_hyd_min, default_hyd_max = -5.0, 5.0
    elif mode == "Sleep":
        default_hr_min, default_hr_max = 40, 100
        default_hyd_min, default_hyd_max = -5.0, 5.0
    else:  # Disorder-safe
        default_hr_min, default_hyd_min = 30, -5.0
        default_hr_max, default_hyd_max = 220, 5.0
    
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

    hyd_min = st.number_input("Hydration min (%)", 
                             value=float(default_hyd_min), 
                             step=1.0, 
                             format="%.1f",
                             help="Below this indicates dehydration")
    
    hyd_max = st.number_input("Hydration max (%)", 
                             value=float(default_hyd_max), 
                             step=1.0, 
                             format="%.1f",
                             help="Above this may indicate overhydration")

    max_delta = st.number_input("Max HR jump (bpm/s)", 
                               value=40.0, 
                               step=1.0,
                               help="Sudden jumps above this are flagged as artifacts")

# User-defined thresholds (this is what you wanted: user chooses min/max)
with st.sidebar.expander("Thresholds", expanded=True):
    hr_min = st.number_input("Heart Rate min (bpm)", value=float(default_hr_min), step=1.0)
    hr_max = st.number_input("Heart Rate max (bpm)", value=float(default_hr_max), step=1.0)

    hyd_min = st.number_input("Hydration min (%)", value=float(default_hyd_min), step=1.0, format="%.1f")
    hyd_max = st.number_input("Hydration max (%)", value=float(default_hyd_max), step=1.0, format="%.1f")

    spo2_min = st.number_input("Blood Oxygen min (%)", value=95.0, step=1.0)
    spo2_max = st.number_input("Blood Oxygen max (%)", value=100.0, step=1.0)

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
    window_s = st.slider("Chart window (seconds)", 60, 900, 60, step=30)  # Make sure this exists

    # Smoothing affects plot appearance only (not logged values)
    smoothing = st.selectbox("Smoothing", ["None", "Moving average (5s)", "Moving average (15s)"], index=0)

    # Optional overlay + marker toggles
    show_flag_markers = st.checkbox("Show outlier/disconnect markers", value=True)
    

# ============================================================
# Encryption Settings (Add this after dataset info in sidebar)
# ============================================================

st.sidebar.markdown("---")
st.sidebar.title("🔒 Security Settings")

# Initialize encryption in session state if not already done
if "encryption_enabled" not in st.session_state:
    st.session_state.encryption_enabled = False

# Encryption toggle
enable_encryption = st.sidebar.checkbox(
    "Enable data encryption", 
    value=st.session_state.encryption_enabled,
    help="Encrypt sensitive health data before saving"
)

# Update session state
st.session_state.encryption_enabled = enable_encryption

if enable_encryption:
    encryption_method = st.sidebar.radio(
        "Encryption method",
        ["File-based key", "Password-based"],
        help="File-based: Uses key file (automatic). Password-based: Requires password"
    )
    
    # Key file path
    key_path = BASE_DIR / "encryption_key.key"
    
    if encryption_method == "Password-based":
        enc_password = st.sidebar.text_input("🔑 Encryption password", type="password")
        if enc_password and len(enc_password) >= 8:
            try:
                if 'user_encryptor' not in st.session_state:
                    st.session_state.user_encryptor = PasswordBasedEncryptor(enc_password)
                st.sidebar.success("✅ Password-based encryption ready")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        elif enc_password:
            st.sidebar.warning("Password must be at least 8 characters")
    else:
        # Initialize file-based encryptor
        try:
            if 'encryptor' not in st.session_state:
                st.session_state.encryptor = HealthDataEncryptor(key_file=key_path)
            
            # Show key status
            if key_path.exists():
                st.sidebar.success(f"✅ Key file loaded: `{key_path.name}`")
            else:
                st.sidebar.info(f"🔑 New key will be created at: `{key_path.name}`")
        except Exception as e:
            st.sidebar.error(f"Encryption error: {e}")
    
    # Initialize transmitter if encryptor exists
    if 'encryptor' in st.session_state and 'transmitter' not in st.session_state:
        st.session_state.transmitter = SecureDataTransmitter(st.session_state.encryptor)

# ============================================================
# BLE Connection Section - WITH UNIQUE KEYS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.title("📡 BLE Connection")

# BLE control buttons - ADD UNIQUE KEYS
col_ble1, col_ble2 = st.sidebar.columns(2)
with col_ble1:
    if st.button("🔌 Connect BLE", use_container_width=True, key="ble_connect_button"):
        if start_ble_receiver():
            st.sidebar.success("BLE receiver started!")
        else:
            st.sidebar.warning("BLE receiver already running")

with col_ble2:
    if st.button("⏸ Disconnect BLE", use_container_width=True, key="ble_disconnect_button"):
        stop_ble_receiver()
        st.sidebar.info("BLE receiver stopped")

# Display BLE status
if ble_state["connected"]:
    st.sidebar.success(f"✅ Connected to: {ble_state['device_name']}")
    st.sidebar.write(f"📱 Address: {ble_state['device_addr']}")
    if ble_state["last_update"]:
        ago = time.time() - ble_state["last_update"]
        if ago < 5:
            st.sidebar.info(f"📊 Last update: {ago:.1f}s ago")
        else:
            st.sidebar.warning(f"⚠️ Last update: {ago:.1f}s ago")
else:
    if ble_state["running"]:
        st.sidebar.info("🔍 Searching for device...")
    else:
        st.sidebar.info("❌ Not connected")
    
    if ble_state["error"]:
        st.sidebar.error(f"Error: {ble_state['error']}")

# BLE Command controls - ADD UNIQUE KEYS
if ble_state["connected"]:
    st.sidebar.markdown("### Device Commands")
    col_cmd1, col_cmd2, col_cmd3 = st.sidebar.columns(3)
    
    with col_cmd1:
        if st.button("🔔 Vibrate ON", use_container_width=True, key="ble_vibrate_on_button"):
            if send_command_to_device("MOTOR_ON"):
                st.sidebar.success("Command sent")
            else:
                st.sidebar.warning("Not connected")
    
    with col_cmd2:
        if st.button("🔕 Vibrate OFF", use_container_width=True, key="ble_vibrate_off_button"):
            if send_command_to_device("MOTOR_OFF"):
                st.sidebar.success("Command sent")
            else:
                st.sidebar.warning("Not connected")
    
    with col_cmd3:
        if st.button("🔄 Reset", use_container_width=True, key="ble_reset_button"):
            if send_command_to_device("RESET"):
                st.sidebar.success("Command sent")
            else:
                st.sidebar.warning("Not connected")

# ============================================================
# Main app - ADD BLE DATA PROCESSING HERE
# ============================================================

# Process BLE data before any other operations
process_ble_data()

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
        spo2_threshold_low=spo2_min,
        spo2_threshold_high=spo2_max,
    )

# Keep monitor thresholds aligned with sidebar every rerun
st.session_state.monitor.hr_threshold_low = float(hr_min)
st.session_state.monitor.hr_threshold_high = float(hr_max)
st.session_state.monitor.hydration_threshold_low = float(hyd_min)
st.session_state.monitor.hydration_threshold_high = float(hyd_max)
st.session_state.monitor.spo2_threshold_low = float(spo2_min)
st.session_state.monitor.spo2_threshold_high = float(spo2_max)

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
    hyd = row["hydration_percent"]
    spo2 = row["spo2_percent"]

    # Missing flags
    hr_missing = int(pd.isna(hr_raw))
    hyd_missing = int(pd.isna(hyd))
    spo2_missing = int(pd.isna(spo2))

    # User min/max range flags
    hr_out_of_range = 0
    if not pd.isna(hr_raw):
        hr_out_of_range = int((float(hr_raw) < float(hr_min)) or (float(hr_raw) > float(hr_max)))

    hyd_out_of_range = 0
    if not pd.isna(hyd):
        hyd_out_of_range = int((float(hyd) < float(hyd_min)) or (float(hyd) > float(hyd_max)))

    spo2_out_of_range = 0
    if not pd.isna(spo2):
        spo2_out_of_range = int((float(spo2) < float(spo2_min)) or (float(spo2) > float(spo2_max)))

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
        "spo2_missing": spo2_missing,
        "hr_out_of_range": hr_out_of_range,
        "hyd_out_of_range": hyd_out_of_range,
        "spo2_out_of_range": spo2_out_of_range,
        "hr_artifact": hr_artifact,
    }

def append_flag_log(live_dt: datetime, idx: int, row, flags: dict):
    # Save one row into the per-sample flag log (timestamp + values + flags)
    st.session_state.flag_log.append(
        {
            "timestamp_live": live_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sample": int(idx),
            "hr_bpm_raw": None if pd.isna(row["hr_bpm_raw"]) else float(row["hr_bpm_raw"]),
            "hydration_percent": None if pd.isna(row["hydration_percent"]) else float(row["hydration_percent"]),
            "spo2_percent": None if pd.isna(row["spo2_percent"]) else float(row["spo2_percent"]),
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
# Encryption Helper Functions
# ============================================================

def save_encrypted_session_data():
    """Save current session data with encryption"""
    if not st.session_state.encryption_enabled:
        st.warning("Encryption is not enabled")
        return None
    
    # Prepare data to save
    session_data = {
        'flag_log': st.session_state.flag_log[-1000:],  # Last 1000 entries
        'event_log': st.session_state.event_log[-500:],  # Last 500 entries
        'timestamp': datetime.now().isoformat(),
        'stream_position': st.session_state.i,
        'user_profile': {
            'age': age,
            'weight': weight,
            'height': height,
            'gender': gender
        },
        'thresholds': {
            'hr_min': hr_min,
            'hr_max': hr_max,
            'hyd_min': hyd_min,
            'hyd_max': hyd_max,
            'spo2_min': spo2_min,
            'spo2_max': spo2_max
        }
    }
    
    try:
        # Use appropriate encryptor
        if 'user_encryptor' in st.session_state:
            encrypted = st.session_state.user_encryptor.encrypt_user_data(session_data)
        elif 'encryptor' in st.session_state:
            encrypted = st.session_state.encryptor.encrypt_data(session_data)
        else:
            st.error("No encryptor initialized")
            return None
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = BASE_DIR / f"session_backup_{timestamp}.enc"
        
        with open(save_path, 'wb') as f:
            f.write(encrypted)
        
        st.success(f"✅ Session saved encrypted to `{save_path.name}`")
        return save_path
    
    except Exception as e:
        st.error(f"Failed to save encrypted data: {e}")
        return None

def load_encrypted_session_data(file_path):
    """Load and decrypt session data"""
    try:
        with open(file_path, 'rb') as f:
            encrypted = f.read()
        
        # Try file-based decryptor first
        if 'encryptor' in st.session_state:
            decrypted = st.session_state.encryptor.decrypt_data(encrypted)
            if decrypted:
                return json.loads(decrypted)
        
        # Try password-based decryptor
        if 'user_encryptor' in st.session_state:
            decrypted = st.session_state.user_encryptor.decrypt_user_data(encrypted)
            if decrypted:
                return decrypted
        
        st.error("Could not decrypt with available encryptors")
        return None
    
    except Exception as e:
        st.error(f"Failed to load encrypted data: {e}")
        return None

def create_secure_download_link(data, filename="secure_data.enc"):
    """Create a secure download link for encrypted data"""
    if not st.session_state.encryption_enabled:
        return None
    
    try:
        # Prepare data packet with metadata
        packet = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Encrypt with transmitter if available
        if 'transmitter' in st.session_state:
            secure_packet = st.session_state.transmitter.prepare_for_transmission(
                packet, include_metadata=True
            )
            return json.dumps(secure_packet)
        
        return None
    
    except Exception as e:
        st.error(f"Failed to create secure download: {e}")
        return None

# ============================================================
# Header (title + running badge)
# ============================================================

title_col, badge_col = st.columns([3, 1])

with title_col:
    # Main page title
    st.markdown("## ❤️💧 Capstone Heart Rate, Hydration & Blood Oxygen Monitor")

with badge_col:
    # Small status pill at top right
    status = "RUNNING ✅" if st.session_state.running else "STOPPED ⏸️"
    st.markdown(f'<div class="pill">{status}</div>', unsafe_allow_html=True)


# ============================================================
# Controls row (Start / Stop / Reset + progress)
# ============================================================

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:
    if st.button("▶ Start", use_container_width=True, key="start_button"):
        # Re-anchor timestamps so the current sample aligns with "now"
        st.session_state.stream_start_dt = datetime.now().replace(microsecond=0) - timedelta(seconds=int(st.session_state.i))
        st.session_state.running = True

with c2:
    if st.button("⏸ Stop", use_container_width=True, key="stop_button"):
        st.session_state.running = False

with c3:
    if st.button("↩ Reset", use_container_width=True, key="reset_button"):
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
            spo2_threshold_low=spo2_min,
            spo2_threshold_high=spo2_max,
        )

        # Reset encryption state (optional - comment out if you want to keep encryption settings)
        if 'encryptor' in st.session_state:
            del st.session_state.encryptor
        if 'user_encryptor' in st.session_state:
            del st.session_state.user_encryptor
        if 'transmitter' in st.session_state:
            del st.session_state.transmitter

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
hyd_used = row["hydration_percent"]

# Blood oxygen value (always raw)
spo2_used = row["spo2_percent"]

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
st.session_state.monitor.process_spo2(live_dt, spo2_used)

# Persistent warnings (hold-based) + monitor states
warnings = st.session_state.monitor.get_active_warnings()
monitor_status = st.session_state.monitor.get_status()


# ============================================================
# Tab 1: Dashboard (metrics + alarms + logs)
# ============================================================

with tab1:
    # Top row: HR, Hydration, Blood Oxygen
    k1, k2, k3 = st.columns(3)

    with k1:
        # Use BLE data if available, otherwise use simulated data
        if ble_state["connected"] and ble_state["heart_rate"] is not None:
            hr_display = f"{ble_state['heart_rate']:.1f}"
            hr_metric = st.metric("HR (bpm)", hr_display)
            st.caption("📡 Live BLE Data")
        else:
            hr_used = choose_hr(row)
            hr_display = "—" if pd.isna(hr_used) else f"{float(hr_used):.1f}"
            hr_metric = st.metric("HR (bpm)", hr_display)

    with k2:
        if ble_state["connected"] and ble_state["hydration"] is not None:
            hyd_display = f"{ble_state['hydration']:.1f}%"
            st.metric("Hydration (%)", hyd_display)
            st.caption("📡 Live BLE Data")
        else:
            hyd_used = row["hydration_percent"]
            hyd_display = "—" if pd.isna(hyd_used) else f"{float(hyd_used):.1f}%"
            st.metric("Hydration (%)", hyd_display)

    with k3:
        if ble_state["connected"] and ble_state["spo2"] is not None:
            spo2_display = f"{ble_state['spo2']:.1f}%"
            st.metric("Blood Oxygen (%)", spo2_display)
            st.caption("📡 Live BLE Data")
        else:
            spo2_used = row["spo2_percent"]
            spo2_display = "—" if pd.isna(spo2_used) else f"{float(spo2_used):.1f}%"
            st.metric("Blood Oxygen (%)", spo2_display)

    # Second row: Timestamp and Battery
    k4, k5 = st.columns(2)

    with k4:
        if ble_state["connected"] and ble_state["uptime"]:
            st.metric("Device Uptime", ble_state["uptime"])
            st.caption("📡 From ESP32")
        else:
            st.metric("Live Timestamp", live_dt.strftime("%Y-%m-%d %H:%M:%S"))

    with k5:
        if ble_state["connected"] and ble_state["battery"] is not None:
            st.metric("Battery", f"{ble_state['battery']}%")
            st.caption("📡 Live BLE Data")
        else:
            battery_pct = get_battery_percent_placeholder()
            st.metric("Battery (%)", "—" if battery_pct is None else f"{int(battery_pct)}%")

    st.markdown("<hr/>", unsafe_allow_html=True)

    if ble_state["connected"]:
        st.info("🟢 **Live BLE Connection Active** - Displaying real sensor data from ESP32 Health Patch")
    else:
        st.warning("🔴 **Simulation Mode** - Displaying pre-recorded dataset. Click 'Connect BLE' to receive live data.")

    # LEFT: alarms + alarm history (more important)
    # RIGHT: flags + flag event log + quick checks
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Persistent alerts (hold-based) are the "real alarms"
        if warnings:
            st.warning("### ⏳ Persistent Alerts \n" + "\n".join([f"- {w}" for w in warnings]))
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
        if dyn_flags["spo2_out_of_range"]:
            active_now.append("🔴 Blood oxygen outside user range")
        if dyn_flags["hr_artifact"]:
            active_now.append("🔴 HR artifact (sudden jump)")
        if dyn_flags["hr_missing"]:
            active_now.append("🔴 HR missing (disconnect)")
        if dyn_flags["hyd_missing"]:
            active_now.append("🔴 Hydration missing (disconnect)")
        if dyn_flags["spo2_missing"]:
            active_now.append("🔴 Blood oxygen missing (disconnect)")

        if active_now:
            st.error("### ⚠️ Active Flags \n" + "\n".join([f"- {x}" for x in active_now]))
        else:
            st.success("### ✅ No active flags ")

        st.markdown("#### Quick checks")
        hr_ok = (not pd.isna(row["hr_bpm_raw"])) and (float(hr_min) <= float(row["hr_bpm_raw"]) <= float(hr_max))
        hyd_ok = (not pd.isna(hyd_used)) and (float(hyd_min) <= float(hyd_used) <= float(hyd_max))
        spo2_ok = (not pd.isna(spo2_used)) and (float(spo2_min) <= float(spo2_used) <= float(spo2_max))

        st.write(f"- HR in user range: **{'Yes' if hr_ok else 'No'}**")
        st.write(f"- Hydration in user range: **{'Yes' if hyd_ok else 'No'}**")
        st.write(f"- Blood oxygen in user range: **{'Yes' if spo2_ok else 'No'}**")
        st.write(f"- Auto-clean display: **{'On' if auto_clean else 'Off'}**")
        st.write(f"- Update interval: **{step}s**")

        motor_state = "⚡ ON" if monitor_status["motor_on"] else "⏹️ OFF"
        st.write(f"- Motor state: **{motor_state}**")

        # Add encryption status
        if st.session_state.encryption_enabled:
            st.write(f"- Encryption: **🔒 Enabled**")
            if 'user_encryptor' in st.session_state:
                st.write(f"- Method: **Password-based**")
            elif 'encryptor' in st.session_state:
                st.write(f"- Method: **File-based key**")
        else:
            st.write(f"- Encryption: **🔓 Disabled**")

        # Add secure save button
        if st.session_state.encryption_enabled:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("💾 Save Encrypted Session", use_container_width=True, key="save_encrypted_button"):
                    save_encrypted_session_data()
            
            with col_s2:
                # File uploader for loading
                uploaded_file = st.file_uploader(
                    "Load Encrypted", 
                    type=['enc'],
                    key="enc_uploader",
                    label_visibility="collapsed"
                )
                if uploaded_file is not None:
                    temp_path = BASE_DIR / "temp_upload.enc"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loaded_data = load_encrypted_session_data(temp_path)
                    if loaded_data:
                        st.success("✅ Data loaded successfully!")
                        # You could restore session state here if needed
                    
                    # Clean up
                    temp_path.unlink()

        # Flag event log moved here (less important than alarms)
        st.markdown("### 🧾 Flag Event Log ")
        log_df = pd.DataFrame(st.session_state.flag_log)
        flag_cols = ["hr_out_of_range", "hyd_out_of_range", "spo2_out_of_range", "hr_artifact", "hr_missing", "hyd_missing", "spo2_missing"]

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
    df_window["hyd_sm"] = apply_smoothing(df_window["hydration_percent"])

    # Pull matching saved flags for markers (so chart markers match what’s logged)
    log_df = pd.DataFrame(st.session_state.flag_log)
    log_window = log_df[(log_df["sample"] >= lo) & (log_df["sample"] <= i)].copy()

    # Convert timestamps back into datetime for plotting markers
    if len(log_window) > 0:
        log_window["timestamp_live"] = pd.to_datetime(log_window["timestamp_live"])

    left, middle, right = st.columns(3)

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

    with middle:
        # Hydration chart (smoothed)
        hy_fig = px.line(
            df_window,
            x="timestamp_live",
            y="hyd_sm",
            title="Hydration (recent)",
            labels={"timestamp_live": "Time", "hyd_sm": "Hydration (%)"},
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

    with right:
        # Blood oxygen chart (smoothed)
        df_window["spo2_sm"] = apply_smoothing(df_window["spo2_percent"])
        spo2_fig = px.line(
            df_window,
            x="timestamp_live",
            y="spo2_sm",
            title="Blood Oxygen (recent)",
            labels={"timestamp_live": "Time", "spo2_sm": "Blood Oxygen (%)"},
        )

        if show_flag_markers and len(log_window) > 0:
            pts3 = log_window[(log_window["spo2_out_of_range"] == 1) | (log_window["spo2_missing"] == 1)].copy()

            if len(pts3) > 0:
                s3 = px.scatter(
                    pts3,
                    x="timestamp_live",
                    y="spo2_percent",
                    hover_data=["spo2_out_of_range", "spo2_missing"],
                ).data[0]
                s3.name = "Flagged blood oxygen points (saved)"
                spo2_fig.add_trace(s3)

        spo2_fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(spo2_fig, use_container_width=True)

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

    # Add secure download section if encryption is enabled
    if st.session_state.encryption_enabled:
        st.subheader("🔒 Secure Export")
        
        # Prepare data for secure export
        export_data = {
            'current_readings': {
                'heart_rate': float(hr_used) if not pd.isna(hr_used) else None,
                'hydration': float(hyd_used) if not pd.isna(hyd_used) else None,
                'blood_oxygen': float(spo2_used) if not pd.isna(spo2_used) else None,
                'timestamp': live_dt.isoformat()
            },
            'recent_flags': st.session_state.flag_log[-50:],
            'alarm_history': st.session_state.monitor.alarm_history[-20:],
            'user_profile': {
                'age': age,
                'weight': weight,
                'height': height,
                'gender': gender
            }
        }
        
        secure_packet = create_secure_download_link(export_data)
        if secure_packet:
            st.download_button(
                "🔐 Download Encrypted Snapshot",
                data=secure_packet.encode('utf-8'),
                file_name=f"health_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.enc",
                mime="application/json",
                use_container_width=True
            )
        
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
        try:
            window_log = log_df[(log_df["sample"] >= max(0, i - int(window_s))) & (log_df["sample"] <= i)].copy()
            
            if not window_log.empty:
                st.download_button(
                    "⬇️ Download current window flag log (CSV)",
                    data=window_log.to_csv(index=False).encode("utf-8"),
                    file_name=f"capstone_window_flag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_window_log_button"
                )
            else:
                st.info("No data available in current window for download")
        except Exception as e:
            st.error(f"Error preparing window log: {e}")

    with cdl2:
        # Download entire session log
        try:
            if not log_df.empty:
                st.download_button(
                    "⬇️ Download full flag log (CSV)",
                    data=log_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"capstone_full_flag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_full_log_button"
                )
            else:
                st.info("No log data available for download")
        except Exception as e:
            st.error(f"Error preparing full log: {e}")

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
