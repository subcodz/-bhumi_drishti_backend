import os
import csv
import pytest
from scripts.clean_and_import_historical_data import (
    clean_state,
    clean_district,
    clean_coordinates,
    clean_date,
    clean_rainfall,
    compute_event_severity,
    process_dataset,
)


def test_clean_state():
    assert clean_state("Aarunachal Pradesh") == "Arunachal Pradesh"
    assert clean_state("arunchal pradesh") == "Arunachal Pradesh"
    assert clean_state("assam") == "Assam"
    assert clean_state("Meghalaya") == "Meghalaya"


def test_clean_district():
    assert clean_district("Kamrup moto") == "Kamrup Metro"
    assert clean_district("kamvup metro") == "Kamrup Metro"
    assert clean_district("west kemeng") == "West Kameng"
    assert clean_district("East Sang") == "East Siang"
    assert clean_district("Siang District") == "Siang"


def test_clean_coordinates_swapped():
    # Lat 28.48, Lon 27.10 -> Swapped
    lat, lon, status = clean_coordinates("92.5", "25.5", "East Khasi Hills", "Meghalaya")
    assert status == "SWAPPED_FIXED"
    assert lat == 25.5
    assert lon == 92.5


def test_clean_coordinates_out_of_bounds_imputed():
    lat, lon, status = clean_coordinates("", "", "West Kameng", "Arunachal Pradesh")
    assert status == "IMPUTED_DISTRICT"
    assert 27.0 <= lat <= 28.0
    assert 92.0 <= lon <= 93.0


def test_clean_date_formats():
    assert clean_date("2023-06-15", "2023") == "2023-06-15"
    assert clean_date("15-06-2023", "2023") == "2023-06-15"
    # OCR day 91 correction
    assert clean_date("2021-05-91", "2021") == "2021-05-19"


def test_clean_rainfall():
    # Hyphen typo
    assert clean_rainfall("47-796006", "Meghalaya", 6) == 47.80
    # Missing decimal point
    assert clean_rainfall("73121885", "Assam", 7) == 73.12
    # Standard valid
    assert clean_rainfall("125.5", "Assam", 6) == 125.5
    # Empty rain imputes realistic monsoon average
    rain_imputed = clean_rainfall("", "Meghalaya", 7)
    assert rain_imputed == 180.0


def test_canonical_dataset_integrity():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "cleaned_landslide_incidents.csv")
    assert os.path.exists(csv_path), "cleaned_landslide_incidents.csv must exist"

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 749, f"Expected 749 records, got {len(rows)}"

    for r in rows:
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        # Bounding box of NER India
        assert 21.5 <= lat <= 30.5, f"Lat {lat} out of range in row {r['incident_id']}"
        assert 88.0 <= lon <= 97.5, f"Lon {lon} out of range in row {r['incident_id']}"

        rain = float(r["rainfall_mm"])
        assert 0.0 <= rain <= 1000.0, f"Rain {rain} invalid in row {r['incident_id']}"

        sev = float(r["computed_severity"])
        assert 0.0 <= sev <= 1.0, f"Severity {sev} out of range in row {r['incident_id']}"
