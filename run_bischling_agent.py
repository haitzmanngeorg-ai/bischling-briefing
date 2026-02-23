import os
import json
import math
import smtplib
from email.mime.text import MIMEText
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo

import requests
from openai import OpenAI

LAT = 47.462000
LON = 13.298167
ELEVATION_M = 1840
TZ_NAME = "Europe/Vienna"
LEVELS = [925, 850, 800, 750, 700, 650]


# -----------------------------
# Helpers: wind math
# -----------------------------
def _speed_dir_to_uv(speed_kmh: float, dir_from_deg: float):
    rad = math.radians(dir_from_deg)
    u = -speed_kmh * math.sin(rad)
    v = -speed_kmh * math.cos(rad)
    return u, v


def _uv_to_speed_dir(u: float, v: float):
    speed = math.sqrt(u * u + v * v)
    dir_from = (math.degrees(math.atan2(-u, -v)) % 360.0)
    return speed, dir_from


def _lerp(z, z1, z2, x1, x2):
    if z2 == z1:
        return x1
    w = (z - z1) / (z2 - z1)
    return x1 + w * (x2 - x1)


def _interp_uv_at_height(target_z_msl: float, z_by_level, u_by_level, v_by_level):
    pts = sorted([(z_by_level[L], L) for L in LEVELS if L in z_by_level])
    for (z1, L1), (z2, L2) in zip(pts, pts[1:]):
        if z1 <= target_z_msl <= z2:
            u = _lerp(target_z_msl, z1, z2, u_by_level[L1], u_by_level[L2])
            v = _lerp(target_z_msl, z1, z2, v_by_level[L1], v_by_level[L2])
            return u, v, (L1, L2), (z1, z2)
    return None


def _pick_hour_index(times, target_hour_local: int):
    hh = f"{target_hour_local:02d}:00"
    for i, t in enumerate(times):
        if t.endswith(hh):
            return i
    return 0


# -----------------------------
# Data fetch + derived metrics
# -----------------------------
def get_bischling_weather(date_str: str, hour_local: int = 6):
    url = "https://api.open-meteo.com/v1/gfs"

    hourly = [
        "temperature_2m",
        "dewpoint_2m",
        "cloudcover",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "precipitation",
        "windspeed_10m",
        "winddirection_10m",
        "windgusts_10m",
        # optional convective
        "cape",
        "cin",
    ]

    for L in LEVELS:
        hourly.append(f"wind_speed_{L}hPa")
        hourly.append(f"wind_direction_{L}hPa")
        hourly.append(f"geopotential_height_{L}hPa")

    params = {
        "latitude": LAT,
        "longitude": LON,
        "elevation": ELEVATION_M,
        "timezone": TZ_NAME,
        "hourly": ",".join(hourly),
        "wind_speed_unit": "kmh",
        "start_date": date_str,
        "end_date": date_str,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    times = data["hourly"]["time"]
    idx = _pick_hour_index(times, hour_local)

    def _get_hourly(key, default=None):
        try:
            return data["hourly"][key][idx]
        except Exception:
            return default

    # Surface
    T = float(_get_hourly("temperature_2m"))
    Td = float(_get_hourly("dewpoint_2m"))
    wind10 = float(_get_hourly("windspeed_10m"))
    wind10_dir = float(_get_hourly("winddirection_10m"))
    gusts = float(_get_hourly("windgusts_10m"))
    cloud = float(_get_hourly("cloudcover"))
    cloud_low = float(_get_hourly("cloudcover_low"))
    cloud_mid = float(_get_hourly("cloudcover_mid"))
    cloud_high = float(_get_hourly("cloudcover_high"))
    precip = float(_get_hourly("precipitation"))

    cape = _get_hourly("cape", None)
    cin = _get_hourly("cin", None)
    cape = float(cape) if cape is not None else None
    cin = float(cin) if cin is not None else None

    # Cloudbase (rough)
    cloud_base_agl = max(0.0, 125.0 * (T - Td))
    cloud_base_msl = ELEVATION_M + cloud_base_agl

    # Upper air
    z_by_level, u_by_level, v_by_level = {}, {}, {}
    for L in LEVELS:
        z = float(_get_hourly(f"geopotential_height_{L}hPa"))
        spd = float(_get_hourly(f"wind_speed_{L}hPa"))
        d = float(_get_hourly(f"wind_direction_{L}hPa"))
        z_by_level[L] = z
        u, v = _speed_dir_to_uv(spd, d)
        u_by_level[L] = u
        v_by_level[L] = v

    uv2000 = _interp_uv_at_height(2000.0, z_by_level, u_by_level, v_by_level)
    uv3000 = _interp_uv_at_height(3000.0, z_by_level, u_by_level, v_by_level)

    upper = {"available": False}
    if uv2000 and uv3000:
        u2, v2, br2, hz2 = uv2000
        u3, v3, br3, hz3 = uv3000
        spd2, dir2 = _uv_to_speed_dir(u2, v2)
        spd3, dir3 = _uv_to_speed_dir(u3, v3)

        upper = {
            "available": True,
            "wind_2000m_kmh": round(spd2, 1),
            "wind_2000m_dir_deg": int(round(dir2, 0)),
            "wind_3000m_kmh": round(spd3, 1),
            "wind_3000m_dir_deg": int(round(dir3, 0)),
            "wind_gradient_2000_to_3000_kmh": round(spd3 - spd2, 1),
        }

    # Föhn heuristic (rough)
    foehn = {"risk": "low", "reason": ""}
    if upper.get("available"):
        d3 = upper["wind_3000m_dir_deg"]
        s3 = upper["wind_3000m_kmh"]
        grad = upper["wind_gradient_2000_to_3000_kmh"]
        dryness = (T - Td)

        if 150 <= d3 <= 240 and s3 >= 30 and grad >= 15 and dryness >= 3:
            foehn["risk"] = "high"
            foehn["reason"] = f"S–SW Höhenwind ({s3} km/h aus {d3}°), Gradient {grad} km/h, trocken (T−Td {dryness:.1f}°C)."
        elif 150 <= d3 <= 240 and s3 >= 25 and grad >= 10:
            foehn["risk"] = "medium"
            foehn["reason"] = f"S–SW Höhenwind Tendenz ({s3} km/h aus {d3}°) und Gradient {grad} km/h."
        else:
            foehn["reason"] = "Keine klare S–SW-Föhn-Signatur in 3000m/Gradient."

    # Overdevelopment heuristic based on CAPE/CIN
    overdev = {"risk": "unknown", "reason": "CAPE/CIN nicht verfügbar."}
    if cape is not None:
        cin_val = cin if cin is not None else None
        if cape >= 800 and (cin_val is None or cin_val > -25):
            overdev = {"risk": "high", "reason": f"CAPE {cape:.0f} J/kg, CIN {cin_val if cin_val is not None else 'n/a'} (geringe Deckelung)."}
        elif cape >= 300:
            overdev = {"risk": "medium", "reason": f"CAPE {cape:.0f} J/kg (Instabilität möglich)."}
        else:
            overdev = {"risk": "low", "reason": f"CAPE {cape:.0f} J/kg (eher stabil)."}

    return {
        "date": date_str,
        "time_local_used": times[idx],
        "surface": {
            "temperature_2m_c": T,
            "dewpoint_2m_c": Td,
            "wind_10m_kmh": wind10,
            "wind_10m_dir_deg": wind10_dir,
            "gusts_10m_kmh": gusts,
            "cloudcover_pct": cloud,
            "cloudcover_low_pct": cloud_low,
            "cloudcover_mid_pct": cloud_mid,
            "cloudcover_high_pct": cloud_high,
            "precipitation_mm": precip,
            "cape": cape,
            "cin": cin,
        },
        "derived": {
            "cloud_base_agl_m_est": round(cloud_base_agl, 0),
            "cloud_base_msl_m_est": round(cloud_base_msl, 0),
        },
        "upper_wind": upper,
        "foehn": foehn,
        "overdevelopment": overdev,
    }


# -----------------------------
# Email delivery
# -----------------------------
def send_email(subject: str, body: str):
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to_raw = os.environ.get("EMAIL_TO", "")
    from_addr = os.environ.get("EMAIL_FROM") or user

    if not host or not user or not pw or not to_raw:
        print("Email not configured (missing SMTP_* or EMAIL_TO).")
        return

    recipients = [x.strip() for x in to_raw.split(",") if x.strip()]
    if not recipients:
        print("EMAIL_TO empty.")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, pw)
        s.sendmail(from_addr, recipients, msg.as_string())


# -----------------------------
# Report building via OpenAI
# -----------------------------
def build_prompt(d0, d1, d2):
    return f"""
Du bist ein spezialisierter Paragliding-Wetteranalyst für Bischling (Werfenweng, AT).
Nutze ausschließlich die gelieferten JSON-Daten. Erfinde keine Werte.

Ausgabeformat:
- HEUTE: ausführlich
- MORGEN: mittel (gleiches, aber kürzer)
- ÜBERMORGEN: kurz (nur Entscheidung + 3 Schlüsselwerte)

MUSS enthalten:
- Wind 10m, Wind 2000m, Wind 3000m, Gradient
- Cloudbase MSL (Arbeitshöhe)
- Overdevelopment-Risiko (Gewitter/Überentwicklung) aus overdevelopment
- Föhn-Risiko aus foehn

Bewertungs-Kerne:
- 3000m Wind > 35 km/h oder Gradient > +20 km/h -> Turbulenz/Scherung Warnung
- Cloudbase MSL wichtig für “Arbeitshöhe”
- Bei CAPE hoch + CIN schwach -> Overdev hoch

HEUTE JSON:
{json.dumps(d0, ensure_ascii=False)}

MORGEN JSON:
{json.dumps(d1, ensure_ascii=False)}

ÜBERMORGEN JSON:
{json.dumps(d2, ensure_ascii=False)}
"""


def make_report():
    tz = ZoneInfo(TZ_NAME)
    now_local = datetime.now(tz)

    # DST-safe gating window: send only around 06:30 local
    if False:
        print(f"Not in send window (local time {now_local:%H:%M}). Exiting.")
        return None

    today = date.today()
    tomorrow = today + timedelta(days=1)
    day2 = today + timedelta(days=2)

    # 06:00 snapshot (very likely available). If you prefer closer to 06:30, set to 7.
    hour_local = 6

    d0 = get_bischling_weather(today.isoformat(), hour_local=hour_local)
    d1 = get_bischling_weather(tomorrow.isoformat(), hour_local=hour_local)
    d2 = get_bischling_weather(day2.isoformat(), hour_local=hour_local)

    client = OpenAI()
    messages = [
        {"role": "system", "content": "Du antwortest auf Deutsch, kompakt, pilotentauglich."},
        {"role": "user", "content": build_prompt(d0, d1, d2)},
    ]

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )

    return resp.choices[0].message.content.strip()


def main():
    report = make_report()
    if not report:
        return

    subject = f"Bischling Paragliding Briefing {date.today().isoformat()}"
    send_email(subject, report)
    print("Sent briefing via Email.")


if __name__ == "__main__":
    main()
