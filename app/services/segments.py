import json
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from geoalchemy2.functions import ST_AsGeoJSON, ST_MakeEnvelope, ST_Intersects

from app.db.models import RoadSegment, SegmentFeature, SegmentRisk
from app.services.risk_engine import evaluate_segment_risk
from app.services.ml_engine import predict_segment_ml_risk, get_trained_model


def recalculate_and_save_risk(db: Session, segment: RoadSegment, feat: SegmentFeature, commit: bool = True) -> SegmentRisk:
    """
    Computes risk metrics for a segment using its feature vector and saves/updates SegmentRisk.
    """
    overall_risk, category, hazards = evaluate_segment_risk(feat, segment)

    risk_obj = db.query(SegmentRisk).filter(SegmentRisk.segment_id == segment.segment_id).first()
    if not risk_obj:
        risk_obj = SegmentRisk(segment_id=segment.segment_id)
        db.add(risk_obj)

    risk_obj.overall_blockage_risk = overall_risk
    risk_obj.risk_category = category
    risk_obj.hazard_flood = hazards["flood"]
    risk_obj.hazard_landslide = hazards["landslide"]
    risk_obj.hazard_road_damage = hazards["road_damage"]
    risk_obj.hazard_congestion = hazards["congestion"]

    if commit:
        db.commit()
        db.refresh(risk_obj)
    return risk_obj


NER_STATE_BOUNDS = {
    "AS": (89.60, 24.10, 96.00, 28.00),
    "ASSAM": (89.60, 24.10, 96.00, 28.00),
    "AR": (91.50, 26.50, 97.50, 29.50),
    "ARUNACHAL PRADESH": (91.50, 26.50, 97.50, 29.50),
    "MN": (93.00, 23.80, 94.80, 25.70),
    "MANIPUR": (93.00, 23.80, 94.80, 25.70),
    "ML": (89.80, 25.00, 92.80, 26.15),
    "MEGHALAYA": (89.80, 25.00, 92.80, 26.15),
    "MZ": (92.20, 21.90, 93.50, 24.50),
    "MIZORAM": (92.20, 21.90, 93.50, 24.50),
    "NL": (93.30, 25.20, 95.30, 27.00),
    "NAGALAND": (93.30, 25.20, 95.30, 27.00),
    "TR": (91.10, 22.90, 92.40, 24.50),
    "TRIPURA": (91.10, 22.90, 92.40, 24.50),
}


def get_road_segments(
    db: Session,
    road_type: Optional[str] = None,
    risk_category: Optional[str] = None,
    min_risk: Optional[float] = None,
    state: Optional[str] = None,
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """
    Fetch road segments with optional filtering by road_type, risk_category, min_risk, state, or bounding box.
    Returns a GeoJSON FeatureCollection containing rule-based risk and XGBoost ML probability.
    """
    query = db.query(
        RoadSegment,
        ST_AsGeoJSON(RoadSegment.geom).label("geojson"),
        SegmentRisk,
        SegmentFeature
    ).outerjoin(SegmentRisk, RoadSegment.segment_id == SegmentRisk.segment_id)\
     .outerjoin(SegmentFeature, RoadSegment.segment_id == SegmentFeature.segment_id)

    if road_type:
        query = query.filter(RoadSegment.road_type == road_type)

    if risk_category:
        query = query.filter(SegmentRisk.risk_category == risk_category.upper())

    if min_risk is not None:
        query = query.filter(SegmentRisk.overall_blockage_risk >= min_risk)

    if state:
        st_key = state.strip().upper()
        if st_key in NER_STATE_BOUNDS:
            b = NER_STATE_BOUNDS[st_key]
            state_envelope = ST_MakeEnvelope(b[0], b[1], b[2], b[3], 4326)
            query = query.filter(ST_Intersects(RoadSegment.geom, state_envelope))

    if min_lon is not None and min_lat is not None and max_lon is not None and max_lat is not None:
        bbox_geom = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
        query = query.filter(ST_Intersects(RoadSegment.geom, bbox_geom))

    results = query.limit(limit).all()

    # Vectorized batch ML inference: eliminates N+1 queries for high performance
    model_bundle = get_trained_model()
    clf = model_bundle.get("model") if model_bundle else None

    vectors = []
    valid_indices = []
    for idx, (segment, geojson_str, risk, feat) in enumerate(results):
        if feat and clf:
            vectors.append([
                float(feat.elevation_m),
                float(feat.slope_deg),
                float(feat.terrain_roughness),
                float(feat.distance_to_river_m),
                float(feat.distance_to_stream_m),
                1.0 if feat.within_flood_zone else 0.0,
                float(feat.rainfall_1h_mm),
                float(feat.rainfall_6h_mm),
                float(feat.rainfall_24h_mm),
                float(feat.rainfall_72h_mm),
                float(feat.flood_events_1y),
                float(feat.landslide_events_1y),
                float(feat.blockages_1y),
                float(feat.road_damage_reports_30d),
                1.0 if feat.construction_active else 0.0,
                float(feat.congestion_ratio),
                float(feat.flood_reports_24h),
                float(feat.landslide_reports_24h),
                float(segment.length_m),
                float(segment.lanes if segment.lanes else 2),
            ])
            valid_indices.append(idx)

    ml_probs_dict = {}
    if vectors and clf:
        import numpy as np
        X_batch = np.array(vectors)
        probs = clf.predict_proba(X_batch)[:, 1]
        for i, val_idx in enumerate(valid_indices):
            p = round(float(probs[i]), 4)
            cat = "CRITICAL" if p >= 0.75 else "HIGH" if p >= 0.55 else "MEDIUM" if p >= 0.30 else "LOW"
            ml_probs_dict[val_idx] = (p, cat)

    features = []
    for idx, (segment, geojson_str, risk, feat) in enumerate(results):
        geometry = json.loads(geojson_str)
        ml_prob, ml_cat = ml_probs_dict.get(idx, (None, None))

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "segment_id": segment.segment_id,
                "osm_way_id": segment.osm_way_id,
                "road_type": segment.road_type,
                "lanes": segment.lanes,
                "surface": segment.surface,
                "bridge": segment.bridge,
                "oneway": segment.oneway,
                "maxspeed": segment.maxspeed,
                "name": segment.name,
                "ref": segment.ref,
                "length_m": round(segment.length_m, 2),
                "overall_blockage_risk": risk.overall_blockage_risk if risk else None,
                "risk_category": risk.risk_category if risk else None,
                "ml_blockage_probability": ml_prob,
                "ml_risk_category": ml_cat,
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_road_segment_by_id(db: Session, segment_id: int) -> Optional[Dict[str, Any]]:
    result = db.query(
        RoadSegment,
        ST_AsGeoJSON(RoadSegment.geom).label("geojson"),
        SegmentRisk
    ).outerjoin(SegmentRisk, RoadSegment.segment_id == SegmentRisk.segment_id)\
     .filter(RoadSegment.segment_id == segment_id).first()

    if not result:
        return None

    segment, geojson_str, risk = result
    geometry = json.loads(geojson_str)

    ml_pred = predict_segment_ml_risk(db, segment_id)
    ml_prob = ml_pred["ml_blockage_probability"] if ml_pred else None
    ml_cat = ml_pred["ml_risk_category"] if ml_pred else None

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "segment_id": segment.segment_id,
            "osm_way_id": segment.osm_way_id,
            "road_type": segment.road_type,
            "lanes": segment.lanes,
            "surface": segment.surface,
            "bridge": segment.bridge,
            "oneway": segment.oneway,
            "maxspeed": segment.maxspeed,
            "name": segment.name,
            "ref": segment.ref,
            "length_m": round(segment.length_m, 2),
            "overall_blockage_risk": risk.overall_blockage_risk if risk else None,
            "risk_category": risk.risk_category if risk else None,
            "ml_blockage_probability": ml_prob,
            "ml_risk_category": ml_cat,
        }
    }


def get_segment_features(db: Session, segment_id: int) -> Optional[Dict[str, Any]]:
    segment = db.query(RoadSegment).filter(RoadSegment.segment_id == segment_id).first()
    if not segment:
        return None

    feat = db.query(SegmentFeature).filter(SegmentFeature.segment_id == segment_id).first()
    if not feat:
        feat = SegmentFeature(segment_id=segment_id)
        db.add(feat)
        db.commit()
        db.refresh(feat)
        recalculate_and_save_risk(db, segment, feat)

    return {
        "segment_id": segment.segment_id,
        "road": {
            "road_type": segment.road_type,
            "lanes": segment.lanes,
            "surface": segment.surface,
            "length_m": round(segment.length_m, 2),
            "bridge": segment.bridge,
            "oneway": segment.oneway,
            "maxspeed": segment.maxspeed,
            "name": segment.name,
            "ref": segment.ref,
        },
        "terrain": {
            "elevation_m": feat.elevation_m,
            "slope_deg": feat.slope_deg,
            "terrain_roughness": feat.terrain_roughness,
        },
        "hydrology": {
            "distance_to_river_m": feat.distance_to_river_m,
            "distance_to_stream_m": feat.distance_to_stream_m,
            "within_flood_zone": feat.within_flood_zone,
        },
        "weather": {
            "rainfall_1h_mm": feat.rainfall_1h_mm,
            "rainfall_6h_mm": feat.rainfall_6h_mm,
            "rainfall_24h_mm": feat.rainfall_24h_mm,
            "rainfall_72h_mm": feat.rainfall_72h_mm,
        },
        "history": {
            "flood_events_1y": feat.flood_events_1y,
            "landslide_events_1y": feat.landslide_events_1y,
            "blockages_1y": feat.blockages_1y,
        },
        "infrastructure": {
            "road_damage_reports_30d": feat.road_damage_reports_30d,
            "construction_active": feat.construction_active,
        },
        "traffic": {
            "congestion_ratio": feat.congestion_ratio,
        },
        "field_reports": {
            "flood_reports_24h": feat.flood_reports_24h,
            "landslide_reports_24h": feat.landslide_reports_24h,
        }
    }


def update_segment_features(db: Session, segment_id: int, update_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    segment = db.query(RoadSegment).filter(RoadSegment.segment_id == segment_id).first()
    if not segment:
        return None

    feat = db.query(SegmentFeature).filter(SegmentFeature.segment_id == segment_id).first()
    if not feat:
        feat = SegmentFeature(segment_id=segment_id)
        db.add(feat)

    for field, value in update_dict.items():
        if value is not None and hasattr(feat, field):
            setattr(feat, field, value)

    db.commit()
    db.refresh(feat)

    recalculate_and_save_risk(db, segment, feat)

    return get_segment_features(db, segment_id)


def get_segment_risk(db: Session, segment_id: int) -> Optional[Dict[str, Any]]:
    segment = db.query(RoadSegment).filter(RoadSegment.segment_id == segment_id).first()
    if not segment:
        return None

    risk = db.query(SegmentRisk).filter(SegmentRisk.segment_id == segment_id).first()
    if not risk:
        feat = db.query(SegmentFeature).filter(SegmentFeature.segment_id == segment_id).first()
        if not feat:
            feat = SegmentFeature(segment_id=segment_id)
            db.add(feat)
            db.commit()
            db.refresh(feat)
        risk = recalculate_and_save_risk(db, segment, feat)

    return {
        "segment_id": segment.segment_id,
        "overall_blockage_risk": risk.overall_blockage_risk,
        "risk_category": risk.risk_category,
        "hazards": {
            "flood": risk.hazard_flood,
            "landslide": risk.hazard_landslide,
            "road_damage": risk.hazard_road_damage,
            "congestion": risk.hazard_congestion,
        }
    }
