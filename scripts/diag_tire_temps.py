"""
Diagnostic: find any iRacing telemetry channel that reports tire temps LIVE.

Run this while sitting on track and driving (NOT in the pits). It enumerates
every telemetry variable whose name relates to tire temp / wear / pressure,
then prints a row only when one of those values actually changes -- tagged with
lap, lap %, and on_pit_road so you can see WHETHER updates happen on-track or
only at spawn / pit road.

If the only rows printed are at pit-road transitions, that confirms iRacing's
tire temp/wear channels are crew-measured (frozen during a stint) and there is
no live channel to read. If some channel updates mid-lap on track, it will show
up here.

Usage:
    python scripts/diag_tire_temps.py
Stop with Ctrl+C.
"""

import time

import irsdk


def find_tire_vars(ir):
    """Return telemetry var names related to tire temp/wear/pressure."""
    names = []
    for name in ir._var_headers_names:
        low = name.lower()
        is_tire = name[:2] in ("LF", "RF", "LR", "RR")
        if is_tire and ("temp" in low or "wear" in low or "press" in low):
            names.append(name)
    return sorted(names)


def main():
    ir = irsdk.IRSDK()
    print("Waiting for iRacing... (start a session and go on track)")
    while not ir.startup() or not ir.is_connected:
        time.sleep(1)

    tire_vars = find_tire_vars(ir)
    print(f"\nFound {len(tire_vars)} tire-related channels:")
    print("  " + ", ".join(tire_vars))
    print("\nDrive on track. A row prints only when a value CHANGES.")
    print("Watch the 'pit' column: if changes ONLY happen at pit/spawn, the")
    print("data is crew-measured (no live channel).\n")
    print(f"{'time':>7} | {'lap':>3} {'lap%':>5} {'pit':>3} | changed channels")
    print("-" * 70)

    t0 = time.time()
    prev = {}
    try:
        while True:
            ir.freeze_var_buffer_latest()
            if not ir.is_connected:
                time.sleep(0.5)
                continue

            lap = ir["Lap"]
            pct = ir["LapDistPct"] or 0.0
            pit = ir["OnPitRoad"]

            changed = []
            for v in tire_vars:
                val = ir[v]
                val = round(val, 2) if isinstance(val, float) else val
                if prev.get(v) != val:
                    changed.append(f"{v}={val}")
                    prev[v] = val

            if changed:
                ts = time.time() - t0
                print(
                    f"{ts:7.1f} | {lap:>3} {pct:5.3f} {str(bool(pit)):>5} | "
                    + ", ".join(changed)
                )

            time.sleep(0.1)  # 10 Hz
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ir.shutdown()


if __name__ == "__main__":
    main()
