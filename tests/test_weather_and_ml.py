import os
import joblib
import numpy as np
import pytest
from app.services.imd_weather import fetch_imd_weather_for_coordinate
from app.services.ml_engine import get_trained_model, MODEL_PATH
from scripts.train_model import FEATURE_NAMES


def test_weather_live_fetch():
    # Test for Guwahati and Shillong
    r1, r6, r24, r72 = fetch_imd_weather_for_coordinate(26.14, 91.73)
    assert isinstance(r1, float)
    assert isinstance(r6, float)
    assert isinstance(r24, float)
    assert isinstance(r72, float)
    assert r1 >= 0.0
    assert r6 >= r1 - 0.1  # 6h accumulation should be >= 1h (accounting for float rounding)
    assert r24 >= r6 - 0.1
    assert r72 >= r24 - 0.1


def test_weather_grid_caching():
    # Calling the same coordinate twice within 30 minutes must hit memory cache instantly
    coords = (25.57, 91.88)
    first_res = fetch_imd_weather_for_coordinate(*coords)
    second_res = fetch_imd_weather_for_coordinate(*coords)
    assert first_res == second_res


def test_trained_model_exists_and_loads():
    assert os.path.exists(MODEL_PATH), "road_risk_model.joblib must exist in models/"
    bundle = get_trained_model()
    assert bundle is not None
    assert "model" in bundle
    assert "feature_names" in bundle
    assert bundle["feature_names"] == FEATURE_NAMES
    assert len(bundle["feature_names"]) == 20


def test_ml_prediction_responsiveness():
    bundle = get_trained_model()
    clf = bundle["model"]

    # Severe monsoon trigger on a steep hill corridor:
    # rainfall_24h=200, rainfall_72h=350, slope=32, landslide_events_1y=2
    x_storm = np.array([[
        1100.0, 32.0, 0.6, 60.0, 30.0, 1.0,
        25.0, 90.0, 200.0, 350.0,
        1.0, 2.0, 3.0, 2.0, 0.0, 0.3,
        1.0, 2.0, 500.0, 2.0
    ]])
    prob_storm = float(clf.predict_proba(x_storm)[0][1])

    # Calm dry weather on a gentle plain:
    # rainfall=0, slope=3, landslide_events=0
    x_dry = np.array([[
        120.0, 3.0, 0.1, 800.0, 400.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.1,
        0.0, 0.0, 500.0, 2.0
    ]])
    prob_dry = float(clf.predict_proba(x_dry)[0][1])

    assert prob_storm > 0.70, f"Storm probability {prob_storm} should be high"
    assert prob_dry < 0.30, f"Dry weather probability {prob_dry} should be low"
    assert prob_storm > prob_dry
