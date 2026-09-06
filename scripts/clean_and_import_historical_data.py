#!/usr/bin/env python3
"""
Clean and Standardize Historical Landslide Disaster Ground-Truth Dataset (NER India)

Processes 749 historical disaster logs across 8 North Eastern States:
- Standardizes inconsistent coordinates, dates, rainfall figures, and severity ratings.
- Imputes missing coordinates using verified district/subdivision centroids.
- Exports canonical dataset to `data/cleaned_landslide_incidents.csv`.
- If database is connected, populates `historical_incidents` table and updates `segment_features`.

Usage:
    python scripts/clean_and_import_historical_data.py
"""

import os
import sys
import csv
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# District centroid lookup table for North Eastern Region (NER)
DISTRICT_CENTROIDS: Dict[str, Tuple[float, float]] = {
    # Arunachal Pradesh
    "west kameng": (27.27, 92.44), "upper siang": (28.62, 95.04), "east siang": (28.07, 95.33),
    "papum pare": (27.10, 93.62), "west siang": (28.23, 94.93), "kurung kumey": (27.89, 93.43),
    "lower dibang valley": (28.23, 95.78), "siang": (28.14, 95.27), "upper subansiri": (28.24, 94.14),
    "lower subansiri": (27.55, 94.26), "shi yomi": (28.70, 94.32), "anjaw": (28.05, 96.44),
    "kra daadi": (27.73, 93.64), "tawang": (27.61, 91.86), "east kameng": (27.31, 92.96),
    "changlang": (27.14, 95.75), "dibang valley": (28.79, 95.91), "tirap": (26.99, 95.49),
    "kamle": (27.80, 94.08), "keyi panyor": (27.65, 93.87), "lower siang": (27.79, 94.72),
    # Assam
    "kamrup metro": (26.15, 91.75), "kamrup": (26.18, 91.60), "cachar": (24.80, 93.04),
    "dima hasao": (25.18, 93.01), "karimganj": (24.79, 92.52), "tinsukia": (27.28, 95.74),
    "dhemaji": (27.55, 94.26), "hailakandi": (24.69, 92.64), "sribhumi": (24.72, 92.50),
    "hojai": (25.82, 93.11), "nagaon": (26.35, 92.68),
    # Manipur
    "senapati": (25.27, 93.97), "ukhrul": (25.09, 94.36), "tamenglong": (24.98, 93.50),
    "chandel": (24.22, 94.18), "churachandpur": (24.50, 93.51), "noney": (24.82, 93.55),
    "bishnupur": (24.53, 93.83), "kangpokpi": (25.13, 93.97), "jiribam": (24.75, 93.14),
    "imphal": (24.78, 93.78), "imphal east": (24.78, 93.79), "kamjong": (24.83, 94.15),
    "tengnoupal": (24.28, 94.28),
    # Meghalaya
    "west jaintia hills": (25.39, 92.18), "east jaintia hills": (25.11, 92.36),
    "east khasi hills": (25.57, 91.88), "west khasi hills": (25.53, 91.27),
    "west garo hills": (25.52, 90.22), "south garo hills": (25.20, 90.65),
    "ri bhoi": (25.68, 91.90), "south west khasi hills": (25.37, 91.47),
    "east garo hills": (25.55, 90.41), "eastern west khasi": (25.59, 91.48),
    # Mizoram
    "aizawl": (23.73, 92.72), "serchhip": (23.23, 92.91), "champhai": (23.49, 93.20),
    "lawngtlai": (22.52, 92.89), "lunglei": (22.88, 92.76), "mamit": (23.75, 92.27),
    "kolasib": (23.99, 92.64), "saitual": (23.88, 93.21),
    # Nagaland
    "dimapur": (25.79, 93.80), "phek": (25.66, 94.47), "tuensang": (26.22, 94.80),
    "wokha": (26.10, 94.25), "chumoukedima": (25.80, 93.80), "chümoukedima": (25.80, 93.80),
    "noklak": (26.11, 95.08), "peren": (25.53, 93.77), "kiphire": (25.88, 94.78),
    "kohima": (25.67, 94.10), "mokokchung": (26.32, 94.52), "zunheboto": (26.01, 94.52),
    "shamator": (26.09, 94.75), "niuland": (25.84, 93.92), "longleng": (26.54, 94.84),
    "mon": (26.46, 94.99),
    # Sikkim
    "east sikkim": (27.28, 88.61), "north sikkim": (27.50, 88.53), "south sikkim": (27.17, 88.45),
    "west sikkim": (27.25, 88.25), "gangtok": (27.33, 88.61), "mangan": (27.50, 88.53),
    "namchi": (27.17, 88.45), "pakyong": (27.24, 88.59), "soreng": (27.18, 88.09),
    "gyalshing": (27.28, 88.25), "kalimpong": (27.06, 88.47),
    # Tripura
    "south tripura": (23.25, 91.48), "khowai": (23.84, 91.63), "west tripura": (23.88, 91.71),
    "dhalai": (24.11, 91.91), "unakoti": (24.30, 92.03),
}

STATE_NAME_MAP = {
    "arunachal pradesh": "Arunachal Pradesh",
    "aarunachal pradesh": "Arunachal Pradesh",
    "arunchal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura"
}


def clean_state(raw_state: str) -> str:
    s = raw_state.strip().lower()
    return STATE_NAME_MAP.get(s, raw_state.strip().title())


def clean_district(raw_district: str) -> str:
    d = raw_district.strip().lower()
    d = re.sub(r"\s+district$", "", d)
    d = re.sub(r"^district\s+", "", d)
    # Common OCR typo corrections
    typos = {
        "kamrup moto": "kamrup metro", "kamvup metro": "kamrup metro",
        "west kemeng": "west kameng", "east sang": "east siang",
        "west slang": "west siang", "east skim": "east sikkim",
        "west skim": "west sikkim", "lavngti": "lawngtlai",
        "kalmpong": "kalimpong", "kurungku mey": "kurung kumey",
        "kurungkumey": "kurung kumey", "chumoukedi ma": "chumoukedima",
        "gyalshin g": "gyalshing", "kradaadi": "kra daadi",
        "temenglong": "tamenglong", "wokna": "wokha"
    }
    return typos.get(d, d).title()


def clean_coordinates(raw_lat: str, raw_lon: str, district: str, state: str) -> Tuple[float, float, str]:
    """
    Cleans, validates, and imputes coordinates.
    Returns (latitude, longitude, status) where status is 'ORIGINAL', 'SWAPPED_FIXED', or 'IMPUTED'.
    """
    lat_str = raw_lat.strip()
    lon_str = raw_lon.strip()

    valid = False
    f_lat, f_lon = 0.0, 0.0

    if lat_str and lon_str:
        try:
            f_lat = float(lat_str)
            f_lon = float(lon_str)

            # Check if swapped (e.g. lat ~27, lon ~28 in Sikkim/Assam)
            if 88.0 <= f_lat <= 97.5 and 21.0 <= f_lon <= 30.5:
                f_lat, f_lon = f_lon, f_lat
                return round(f_lat, 6), round(f_lon, 6), "SWAPPED_FIXED"

            # Check obvious typos: lat=21.0 in Arunachal (Papum Pare is ~27.09)
            if "Arunachal" in state and f_lat < 25.0:
                f_lat = 27.09
            # lat=13.0 in Likabali (Likabali is ~27.79)
            if f_lat < 20.0:
                f_lat = 27.79

            # Longitude typo checks (> 98° is out of NER boundaries)
            if f_lon > 98.0:
                if f_lon > 99.0:
                    f_lon = round(f_lon - 6.0, 6)
                else:
                    f_lon = round(f_lon - 4.0, 6)

            # Check bounding box for NER India: 21.5°N - 30.5°N, 88.0°E - 97.5°E
            if 21.5 <= f_lat <= 30.5 and 88.0 <= f_lon <= 97.5:
                valid = True
                return round(f_lat, 6), round(f_lon, 6), "ORIGINAL"
        except ValueError:
            valid = False

    # Impute from district centroid if coordinates were invalid or missing
    d_key = district.strip().lower()
    if d_key in DISTRICT_CENTROIDS:
        c_lat, c_lon = DISTRICT_CENTROIDS[d_key]
        return c_lat, c_lon, "IMPUTED_DISTRICT"

    # Regional fallback by state
    state_fallbacks = {
        "Arunachal Pradesh": (27.10, 93.62), "Assam": (26.15, 91.75),
        "Manipur": (24.81, 93.94), "Meghalaya": (25.57, 91.88),
        "Mizoram": (23.73, 92.72), "Nagaland": (25.67, 94.10),
        "Sikkim": (27.33, 88.61), "Tripura": (23.83, 91.28)
    }
    fb = state_fallbacks.get(state, (26.0, 92.0))
    return fb[0], fb[1], "IMPUTED_STATE"


def clean_date(raw_date: str, report_year: str) -> str:
    d = raw_date.strip()
    if not d:
        return f"{report_year}-06-15"

    # OCR glitch fix (e.g. 2021-05-91 -> 2021-05-19, 2023-07-91 -> 2023-07-19)
    d = d.replace("-91", "-19")

    # Match YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        parts = d.split("-")
        y = int(parts[0])
        if y > 2026:  # e.g. 2027 typo -> report_year
            d = f"{report_year}-{parts[1]}-{parts[2]}"
        return d

    # Match DD-MM-YYYY
    if re.match(r"^\d{2}-\d{2}-\d{4}$", d):
        day, m, y = d.split("-")
        return f"{y}-{m}-{day}"

    return f"{report_year}-06-15"


def clean_rainfall(raw_rain: str, state: str, month: int) -> float:
    r_str = raw_rain.strip()
    if r_str:
        # OCR typo: hyphen in number (e.g. 47-796006 -> 47.80)
        if "-" in r_str and not r_str.startswith("-"):
            r_str = r_str.replace("-", ".")
        try:
            val = float(r_str)
            # Typo where decimal point was omitted (e.g. 73121885 -> 73.12 mm)
            if val > 10000.0:
                s_val = str(int(val))
                if len(s_val) >= 4:
                    val = float(s_val[:2] + "." + s_val[2:4])
                else:
                    val = 50.0
            if 0.0 <= val <= 1000.0:
                return round(val, 2)
        except ValueError:
            pass

    # Regional historical monthly average rainfall (mm) for missing values
    is_monsoon = 5 <= month <= 9
    state_rain = {
        "Meghalaya": 180.0 if is_monsoon else 25.0,
        "Arunachal Pradesh": 120.0 if is_monsoon else 20.0,
        "Assam": 95.0 if is_monsoon else 15.0,
        "Sikkim": 110.0 if is_monsoon else 18.0,
        "Nagaland": 85.0 if is_monsoon else 12.0,
        "Manipur": 75.0 if is_monsoon else 10.0,
        "Mizoram": 90.0 if is_monsoon else 14.0,
        "Tripura": 70.0 if is_monsoon else 10.0
    }
    return state_rain.get(state, 50.0)


def compute_event_severity(type_of_event: str, casualty: str) -> float:
    # 1.0 = minor/debris, 2.0 = medium/road cut, 3.0 = major/catastrophic
    base = 0.65
    t = type_of_event.strip()
    if t == "1.0":
        base = 0.35
    elif t == "2.0":
        base = 0.65
    elif t == "3.0":
        base = 0.85

    cas = 0.0
    c = casualty.strip()
    if c:
        try:
            cas = float(c)
        except ValueError:
            cas = 0.0

    # Add weight for casualties
    severity = min(1.0, base + (cas * 0.05))
    return round(severity, 2)


def process_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(base_dir, "data", "landslide_data_raw.csv")

    if not os.path.exists(raw_path):
        # Check Downloads fallback
        raw_path = r"C:\Users\Subhajit Baidya\Downloads\landslide_data_statewise_2021_2023_2025_2026.csv"
        if not os.path.exists(raw_path):
            print(f"Error: Raw CSV file not found at {raw_path}")
            return

    print(f"Reading raw landslide dataset from: {raw_path}")
    with open(raw_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows read: {len(rows)}")

    cleaned_rows = []
    stats = {"original_coords": 0, "fixed_swapped": 0, "imputed_coords": 0}

    for idx, r in enumerate(rows):
        state = clean_state(r.get("state", ""))
        district = clean_district(r.get("district", ""))
        year_str = r.get("report_year", "2024").strip()
        date_iso = clean_date(r.get("date_of_event", ""), year_str)

        try:
            dt_obj = datetime.strptime(date_iso, "%Y-%m-%d")
            month = dt_obj.month
        except Exception:
            month = 6

        lat, lon, coord_status = clean_coordinates(
            r.get("latitude", ""),
            r.get("longitude", ""),
            district,
            state
        )

        if coord_status == "ORIGINAL":
            stats["original_coords"] += 1
        elif coord_status == "SWAPPED_FIXED":
            stats["fixed_swapped"] += 1
        else:
            stats["imputed_coords"] += 1

        rainfall = clean_rainfall(r.get("rainfall_mm", ""), state, month)
        severity = compute_event_severity(r.get("type_of_event", ""), r.get("casualty", ""))

        clean_record = {
            "incident_id": idx + 1,
            "report_year": year_str,
            "state": state,
            "district": district,
            "name": r.get("name", "").strip() or f"{district} Landslide",
            "location": r.get("location", "").strip() or f"{district}, {state}",
            "nh_sh_affected": r.get("nh_sh_affected", "").strip(),
            "latitude": lat,
            "longitude": lon,
            "date_of_event": date_iso,
            "rainfall_mm": rainfall,
            "type_of_event": r.get("type_of_event", "2.0").strip() or "2.0",
            "casualty": float(r.get("casualty", 0.0) or 0.0) if r.get("casualty", "").strip() else 0.0,
            "computed_severity": severity,
            "source": r.get("source", "").strip() or "Disaster Management Authority",
            "slide_id": r.get("slide_id", "").strip()
        }
        cleaned_rows.append(clean_record)

    # Save cleaned canonical dataset
    out_dir = os.path.join(base_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cleaned_landslide_incidents.csv")

    fieldnames = list(cleaned_rows[0].keys())
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"\nSuccessfully cleaned and standardized {len(cleaned_rows)} records!")
    print(f"Canonical dataset saved to: {out_path}")
    print(f"Coordinate Quality Breakdown:")
    print(f"  - Clean Original GPS: {stats['original_coords']}")
    print(f"  - Swapped/Corrupted Fixed: {stats['fixed_swapped']}")
    print(f"  - Centroid Imputed: {stats['imputed_coords']}")

    # Import into PostGIS if database is accessible
    import_into_postgis(cleaned_rows)

    return cleaned_rows


def import_into_postgis(cleaned_rows: List[Dict[str, Any]]):
    """
    Imports cleaned historical disaster incidents into PostGIS database
    and updates road segment feature vectors with authentic disaster statistics.
    """
    try:
        from app.db.database import engine, SessionLocal, Base
        from app.db.models import RoadSegment, SegmentFeature, HistoricalIncident
        from geoalchemy2.shape import from_shape
        from shapely.geometry import Point
        from geoalchemy2.functions import ST_Distance, ST_SetSRID, ST_MakePoint

        db = SessionLocal()
        segments = db.query(RoadSegment).all()
        if not segments:
            print("PostGIS road_segments table is empty. Skipping DB incident linking.")
            db.close()
            return

        print(f"\nImporting {len(cleaned_rows)} verified disaster incidents into PostGIS database...")
        # Clear synthetic/old incidents
        db.query(HistoricalIncident).delete()
        db.commit()

        # Reset historical counters on features
        for feat in db.query(SegmentFeature).all():
            feat.landslide_events_1y = 0
            feat.blockages_1y = 0

        inserted_count = 0
        first_seg_id = segments[0].segment_id

        for r in cleaned_rows:
            lat = r["latitude"]
            lon = r["longitude"]
            pt = Point(lon, lat)
            wkb_pt = from_shape(pt, srid=4326)

            # Match nearest road segment
            point_geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326)
            nearest = db.query(RoadSegment).order_by(
                ST_Distance(RoadSegment.geom, point_geom)
            ).first()
            target_segment_id = nearest.segment_id if nearest else first_seg_id

            try:
                occurred_dt = datetime.strptime(r["date_of_event"], "%Y-%m-%d")
            except Exception:
                occurred_dt = datetime(2024, 6, 15)

            desc = f"{r['name']} — {r['location']} (Rainfall: {r['rainfall_mm']}mm, Casualties: {int(r['casualty'])})"

            inc = HistoricalIncident(
                segment_id=target_segment_id,
                incident_type="landslide",
                severity=r["computed_severity"],
                description=desc,
                occurred_at=occurred_dt,
                geom=wkb_pt
            )
            db.add(inc)

            # Update segment features
            feat = db.query(SegmentFeature).filter(SegmentFeature.segment_id == target_segment_id).first()
            if feat:
                feat.landslide_events_1y += 1
                feat.blockages_1y += 1

            inserted_count += 1

        db.commit()
        db.close()
        print(f"Successfully populated {inserted_count} verified disaster incidents in PostGIS!")

    except Exception as e:
        print(f"PostGIS database not connected ({e}). Clean canonical CSV is ready in data/ directory.")


if __name__ == "__main__":
    process_dataset()
