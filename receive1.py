"""
receive.py – BLE client for the ESP32 Health-Patch
Displays a live terminal dashboard of all sensor data.

Usage:
    python receive.py

Requirements:
    pip install bleak
"""

import asyncio
import sys
import time
import threading
import queue
from datetime import datetime
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
__all__ = [
    'find_device',
    'CHARACTERISTIC_UUID',
    'HEART_RATE_CHAR_UUID', 
    'HYDRATION_CHAR_UUID',
    'BATTERY_CHAR_UUID',
    'WARNING_CHAR_UUID',
    'COMBINED_SERVICE_UUID',
    'DEVICE_NAME'
]

CHARACTERISTIC_UUID   = "0000abce-0000-1000-8000-00805f9b34fb"
HEART_RATE_CHAR_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
HYDRATION_CHAR_UUID   = "abcdef04-1234-5678-9abc-def012345678"
BATTERY_CHAR_UUID     = "00002a19-0000-1000-8000-00805f9b34fb"
WARNING_CHAR_UUID     = "abcdef02-1234-5678-9abc-def012345678"
COMBINED_SERVICE_UUID = "0000abcd-0000-1000-8000-00805f9b34fb"
DEVICE_NAME           = "Health-Patch"

SCAN_TIMEOUT = 8.0

# ---------------------------------------------------------------------------
# Shared sensor state – written by BLE callbacks, read by the renderer
# ---------------------------------------------------------------------------
state = {
    "heart_rate":   None,
    "hydration":    None,
    "battery":      None,
    "uptime":       None,
    "last_update":  None,
    "connected":    False,
    "device_name":  "",
    "device_addr":  "",
    "log":          [],
}
MAX_LOG = 6

# Thread-safe queue for commands typed at the keyboard
cmd_queue: queue.Queue = queue.Queue()


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    state["log"].append((ts, msg))
    if len(state["log"]) > MAX_LOG:
        state["log"].pop(0)


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------
def bar(value: float, max_val: float, width: int = 22) -> str:
    filled = max(0, min(width, int(round(value / max_val * width))))
    return "█" * filled + "░" * (width - filled)


def render_dashboard():
    ESC   = "\033["
    CLEAR = "\033[2J\033[H"
    RST   = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    GRN   = "\033[92m"
    YLW   = "\033[93m"
    RED   = "\033[91m"
    CYN   = "\033[96m"
    WHT   = "\033[97m"
    W     = 54

    def div(): return DIM + "─" * W + RST

    def metric_row(label, value_str, color, bar_str, pct_str, zone_str=""):
        lines = []
        lines.append(f"  {BOLD}{WHT}{label}{RST}")
        lines.append(f"  {color}{BOLD}{value_str:<12}{RST}  {zone_str}")
        lines.append(f"  {color}{bar_str}{RST}  {pct_str}")
        return lines

    out = [CLEAR]

    # Header
    out.append(BOLD + CYN + "  ╔" + "═"*(W-4) + "╗" + RST)
    out.append(BOLD + CYN + "  ║   ESP32 Health Patch  –  Live Monitor" + " "*(W-42) + "║" + RST)
    out.append(BOLD + CYN + "  ╚" + "═"*(W-4) + "╝" + RST)
    out.append("")

    # Connection
    if state["connected"]:
        out.append(f"  {GRN}● CONNECTED{RST}  {DIM}{state['device_name']}  {state['device_addr']}{RST}")
    else:
        out.append(f"  {RED}○ DISCONNECTED{RST}")

    if state["last_update"]:
        ago = time.time() - state["last_update"]
        color = GRN if ago < 5 else YLW if ago < 15 else RED
        out.append(f"  {DIM}Last data: {color}{ago:.1f}s ago{RST}")

    out.append("")
    out.append(div())
    out.append("")

    # Heart Rate
    hr = state["heart_rate"]
    if hr is not None:
        c = GRN if 60 <= hr <= 100 else YLW if hr <= 130 else RED
        zone = ("Rest   " if hr < 60 else
                "Normal " if hr < 100 else
                "Cardio " if hr < 140 else "Peak   ")
        out += metric_row("❤  Heart Rate",
                          f"{hr:.1f} BPM", c,
                          bar(hr, 180), f"{hr/180*100:.0f}%", zone)
    else:
        out.append(f"  {BOLD}{WHT}❤  Heart Rate{RST}")
        out.append(f"  {DIM}Waiting for data...{RST}")
        out.append("")

    out.append("")

    # Hydration
    hyd = state["hydration"]
    if hyd is not None:
        c = GRN if hyd >= 60 else YLW if hyd >= 45 else RED
        label = "Good   " if hyd >= 60 else "Fair   " if hyd >= 45 else "Low    "
        out += metric_row("💧 Hydration",
                          f"{hyd:.1f} %", c,
                          bar(hyd, 100), f"{hyd:.0f}%", label)
    else:
        out.append(f"  {BOLD}{WHT}💧 Hydration{RST}")
        out.append(f"  {DIM}Waiting for data...{RST}")
        out.append("")

    out.append("")

    # Battery
    bat = state["battery"]
    if bat is not None:
        c = GRN if bat > 60 else YLW if bat > 20 else RED
        icons = "▓▓▓" if bat > 60 else "▓▓░" if bat > 30 else "▓░░" if bat > 10 else "░░░"
        out += metric_row("🔋 Battery",
                          f"{bat} %", c,
                          bar(bat, 100), f"{bat}%", f"[{icons}]")
    else:
        out.append(f"  {BOLD}{WHT}🔋 Battery{RST}")
        out.append(f"  {DIM}Waiting for data...{RST}")
        out.append("")

    out.append("")

    # Uptime
    out.append(f"  {BOLD}{WHT}⏱  Device Uptime{RST}")
    out.append(f"  {CYN}{BOLD}{state['uptime'] or '—'}{RST}")
    out.append("")

    # Log
    out.append(div())
    out.append(f"  {BOLD}Event Log{RST}")
    for ts, msg in state["log"]:
        out.append(f"  {DIM}{ts}  {RST}{msg}")
    if not state["log"]:
        out.append(f"  {DIM}No events yet.{RST}")

    out.append("")
    out.append(div())
    out.append(f"  {DIM}Commands: on | off | reset | quit{RST}")
    out.append("")

    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data parsers  (called from BLE notification callbacks on the asyncio thread)
# ---------------------------------------------------------------------------
def parse_combined(sender, data: bytearray):
    """Parses: Uptime:HH:MM:SS,HR:72.5,Hyd:51.2%"""
    try:
        text = data.decode("utf-8").strip()
        state["last_update"] = time.time()
        for part in text.split(","):
            part = part.strip()
            if part.startswith("Uptime:"):
                state["uptime"] = part[7:]
            elif part.startswith("HR:"):
                state["heart_rate"] = float(part[3:])
            elif part.startswith("Hyd:"):
                state["hydration"] = float(part[4:].rstrip("%"))
    except UnicodeDecodeError:
        # Data is raw bytes – shouldn't happen for combined char but log it
        log(f"Combined char sent raw bytes: {data.hex()} – check firmware")
    except Exception as exc:
        log(f"Parse error (combined): {exc}")


def parse_heart_rate(sender, data: bytearray):
    """
    Handles two formats the ESP32 may send:
      - Standard BLE HR packet: 2 bytes, [flags, BPM_uint8]
      - Raw little-endian float: 4 bytes (NimBLE float overload)
    """
    try:
        import struct
        if len(data) == 2:
            # Standard BLE HR measurement format
            state["heart_rate"]  = float(data[1])
            state["last_update"] = time.time()
        elif len(data) == 4:
            # NimBLE sent a raw float (setValue(float) overload)
            val = struct.unpack("<f", data)[0]
            if 0 < val < 300:  # sanity check – ignore noise values
                state["heart_rate"]  = val
                state["last_update"] = time.time()
        # If neither format matches, ignore silently
    except Exception as exc:
        log(f"Parse error (HR): {exc}")


def parse_hydration(sender, data: bytearray):
    try:
        state["hydration"]   = float(data.decode("utf-8").strip().rstrip("%"))
        state["last_update"] = time.time()
    except Exception as exc:
        log(f"Parse error (hydration): {exc}")


def parse_battery(sender, data: bytearray):
    try:
        if len(data) >= 1:
            state["battery"]     = int(data[0])
            state["last_update"] = time.time()
    except Exception as exc:
        log(f"Parse error (battery): {exc}")


# ---------------------------------------------------------------------------
# Stdin reader – runs in a plain OS thread so it NEVER blocks the asyncio
# event loop. Commands go into cmd_queue for the async handler to pick up.
# ---------------------------------------------------------------------------
def stdin_reader_thread(stop_flag: threading.Event):
    """Blocking readline loop in a background thread."""
    while not stop_flag.is_set():
        try:
            line = sys.stdin.readline()
            if line:
                cmd_queue.put(line.strip().lower())
        except (EOFError, OSError):
            break


# ---------------------------------------------------------------------------
# Command dispatcher  (async, drains cmd_queue)
# ---------------------------------------------------------------------------
async def command_dispatcher(client: BleakClient, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if cmd in ("quit", "exit"):
            stop_event.set()
        elif cmd in ("1", "on"):
            await send_command(client, "MOTOR_ON")
            log("Sent: MOTOR_ON")
        elif cmd in ("0", "off"):
            await send_command(client, "MOTOR_OFF")
            log("Sent: MOTOR_OFF")
        elif cmd == "reset":
            await send_command(client, "RESET")
            log("Sent: RESET")
        elif cmd:
            log(f"Unknown: '{cmd}'")


async def send_command(client: BleakClient, command: str):
    try:
        await client.write_gatt_char(WARNING_CHAR_UUID, command.encode("utf-8"))
    except Exception as exc:
        log(f"Send error: {exc}")


# ---------------------------------------------------------------------------
# Dashboard refresh loop
# ---------------------------------------------------------------------------
async def dashboard_loop(stop_event: asyncio.Event):
    while not stop_event.is_set():
        render_dashboard()
        await asyncio.sleep(0.25)


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------
async def keepalive(client: BleakClient, stop_event: asyncio.Event):
    while not stop_event.is_set():
        if not client.is_connected:
            state["connected"] = False
            log("Device disconnected")
            stop_event.set()
            break
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
async def find_device() -> BLEDevice | None:
    found_device: BLEDevice | None = None
    found_event  = asyncio.Event()
    seen: dict[str, tuple[BLEDevice, AdvertisementData]] = {}
    target_uuid  = COMBINED_SERVICE_UUID.lower()

    def on_device(device: BLEDevice, adv: AdvertisementData):
        nonlocal found_device
        seen[device.address] = (device, adv)
        if found_event.is_set():
            return
        name  = device.name or adv.local_name or ""
        uuids = [str(u).lower() for u in adv.service_uuids]
        if DEVICE_NAME.lower() in name.lower() or target_uuid in uuids:
            found_device = device
            found_event.set()

    print(f"Scanning for '{DEVICE_NAME}' (up to {SCAN_TIMEOUT:.0f} s)...")
    async with BleakScanner(detection_callback=on_device):
        try:
            await asyncio.wait_for(found_event.wait(), timeout=SCAN_TIMEOUT)
        except asyncio.TimeoutError:
            pass

    if found_device:
        return found_device

    if not seen:
        print("[WARN] No BLE devices found. Check Bluetooth is enabled.")
        return None

    print(f"\n[DIAG] '{DEVICE_NAME}' not found. Visible devices:")
    print(f"  {'Address':<20} {'Name':<24} {'RSSI':>5}  UUIDs")
    print(f"  {'-'*20} {'-'*24} {'-'*5}  {'-'*40}")
    for _addr, (dev, adv) in sorted(seen.items(),
                                    key=lambda x: x[1][1].rssi or -999,
                                    reverse=True):
        name  = (dev.name or adv.local_name or "—")[:24]
        rssi  = adv.rssi if adv.rssi else "?"
        uuids = ", ".join(str(u) for u in adv.service_uuids) or "—"
        print(f"  {dev.address:<20} {name:<24} {str(rssi):>5}  {uuids}")
    print()
    return None


# ---------------------------------------------------------------------------
# Connect and run
# ---------------------------------------------------------------------------
async def connect_and_run(device: BLEDevice, stop_flag: threading.Event):
    try:
        async with BleakClient(device, timeout=15.0) as client:
            state["connected"]   = True
            state["device_name"] = device.name or device.address
            state["device_addr"] = device.address
            log(f"Connected to {device.name or device.address}")

            await asyncio.sleep(1)  # let GATT discovery finish

            # Subscribe to all notifying characteristics
            subs = [
                (CHARACTERISTIC_UUID,  parse_combined,   "Combined"),
                (HEART_RATE_CHAR_UUID, parse_heart_rate, "Heart Rate"),
                (HYDRATION_CHAR_UUID,  parse_hydration,  "Hydration"),
                (BATTERY_CHAR_UUID,    parse_battery,    "Battery"),
            ]
            subscribed = []
            for uuid, handler, label in subs:
                try:
                    await client.start_notify(uuid, handler)
                    subscribed.append(uuid)
                    log(f"Subscribed: {label}")
                except Exception as exc:
                    log(f"No notify for {label}: {exc}")

            # Initial read so dashboard shows values before first notify fires
            for uuid, handler, _ in subs:
                try:
                    handler(None, bytearray(await client.read_gatt_char(uuid)))
                except Exception:
                    pass

            stop_event = asyncio.Event()

            # Run dashboard, command dispatcher, and keepalive concurrently.
            # stdin_reader_thread is already running – command_dispatcher
            # drains its queue without blocking any of these tasks.
            await asyncio.gather(
                dashboard_loop(stop_event),
                command_dispatcher(client, stop_event),
                keepalive(client, stop_event),
            )

            for uuid in subscribed:
                try:
                    await client.stop_notify(uuid)
                except Exception:
                    pass

            state["connected"] = False

    except asyncio.TimeoutError:
        log("Connection timed out")
    except Exception as exc:
        log(f"Connection error: {exc}")
        state["connected"] = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    sys.stdout.write("\033[?25l")  # hide cursor
    sys.stdout.flush()

    # Start stdin reader thread once; it lives for the lifetime of the script
    stop_flag = threading.Event()
    t = threading.Thread(target=stdin_reader_thread, args=(stop_flag,), daemon=True)
    t.start()

    try:
        while True:
            device = await find_device()
            if device is None:
                print("Retrying in 2 s...\n")
                await asyncio.sleep(0)
                continue

            await connect_and_run(device, stop_flag)

            state["connected"] = False
            render_dashboard()
            print("\033[?25h")  # restore cursor briefly
            print("Reconnecting in 3 s...")
            await asyncio.sleep(3)
            sys.stdout.write("\033[?25l")

    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n")
        sys.exit(0)
