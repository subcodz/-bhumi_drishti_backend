import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from geoalchemy2.shape import from_shape, to_shape
from geoalchemy2.functions import ST_AsGeoJSON, ST_Distance
from shapely.geometry import Point

from app.db.database import get_db
from app.db.models import RoadSegment, SegmentFeature, FieldReport
from app.schemas.segment import (
    FieldReportCreateRequest,
    FieldReportStatusUpdate,
    FieldReportResponse,
    FieldReportListResponse
)
from app.services.segments import recalculate_and_save_risk

router = APIRouter(prefix="/reports", tags=["field reports"])


def format_report_response(report: FieldReport, geojson_str: Optional[str] = None) -> FieldReportResponse:
    lat, lon = None, None
    if geojson_str:
        g = json.loads(geojson_str)
        coords = g.get("coordinates", [])
        if coords and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
    elif report.geom is not None:
        try:
            pt = to_shape(report.geom)
            lon, lat = pt.x, pt.y
        except Exception:
            pass

    return FieldReportResponse(
        report_id=report.report_id,
        segment_id=report.segment_id,
        reporter_name=report.reporter_name,
        reporter_role=report.reporter_role,
        report_type=report.report_type,
        severity=report.severity,
        status=report.status,
        description=report.description,
        photo_url=report.photo_url,
        latitude=lat,
        longitude=lon,
        created_at=str(report.created_at),
        updated_at=str(report.updated_at),
    )


@router.post("", response_model=FieldReportResponse)
def submit_field_report(
    report_in: FieldReportCreateRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a real-time field incident report for the Helper Dashboard.
    If GPS coordinates are provided without a segment_id, PostGIS auto-matches the nearest road segment.
    Updates segment features and recalculates risk in real-time.
    """
    target_segment_id = report_in.segment_id
    point_geom = None

    if report_in.latitude is not None and report_in.longitude is not None:
        pt = Point(report_in.longitude, report_in.latitude)
        point_geom = from_shape(pt, srid=4326)

        # Auto-match nearest segment if segment_id was not provided
        if not target_segment_id:
            nearest_seg = db.query(RoadSegment).order_by(
                ST_Distance(RoadSegment.geom, point_geom)
            ).first()
            if nearest_seg:
                target_segment_id = nearest_seg.segment_id

    if target_segment_id:
        seg = db.query(RoadSegment).filter(RoadSegment.segment_id == target_segment_id).first()
        if not seg and report_in.segment_id:
            raise HTTPException(status_code=404, detail=f"Road segment with id {report_in.segment_id} not found")

    new_report = FieldReport(
        segment_id=target_segment_id,
        reporter_name=report_in.reporter_name or "Field Responder",
        reporter_role=report_in.reporter_role or "Patrol Officer",
        report_type=report_in.report_type.lower(),
        severity=report_in.severity,
        status="SUBMITTED",
        description=report_in.description,
        photo_url=report_in.photo_url,
        geom=point_geom,
    )

    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    # Increment segment feature 24h count and recalculate risk
    if target_segment_id:
        feat = db.query(SegmentFeature).filter(SegmentFeature.segment_id == target_segment_id).first()
        if not feat:
            feat = SegmentFeature(segment_id=target_segment_id)
            db.add(feat)

        rtype = report_in.report_type.lower()
        if rtype == "flood":
            feat.flood_reports_24h += 1
        elif rtype == "landslide":
            feat.landslide_reports_24h += 1
        elif rtype in ("road_damage", "damage"):
            feat.road_damage_reports_30d += 1

        db.commit()
        db.refresh(feat)

        segment = db.query(RoadSegment).filter(RoadSegment.segment_id == target_segment_id).first()
        if segment:
            recalculate_and_save_risk(db, segment, feat)

    return format_report_response(new_report)


@router.get("", response_model=FieldReportListResponse)
def list_field_reports(
    status: Optional[str] = Query(None, description="Filter by status (SUBMITTED, UNDER_REVIEW, VERIFIED, RESOLVED)"),
    report_type: Optional[str] = Query(None, description="Filter by type (flood, landslide, road_damage, blockage)"),
    segment_id: Optional[int] = Query(None, description="Filter by segment_id"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    List field reports for the dashboard table with optional status and type filtering.
    """
    query = db.query(FieldReport, ST_AsGeoJSON(FieldReport.geom).label("geojson"))

    if status:
        query = query.filter(FieldReport.status == status.upper())
    if report_type:
        query = query.filter(FieldReport.report_type == report_type.lower())
    if segment_id:
        query = query.filter(FieldReport.segment_id == segment_id)

    query = query.order_by(FieldReport.created_at.desc())
    total_count = query.count()
    results = query.limit(limit).all()

    formatted_list = [
        format_report_response(report, geojson_str) for report, geojson_str in results
    ]

    # If no citizen reports submitted yet, provide verified ground-truth incidents
    if total_count == 0:
        from app.db.models import HistoricalIncident
        incidents = db.query(
            HistoricalIncident,
            ST_AsGeoJSON(HistoricalIncident.geom).label("geojson")
        ).order_by(HistoricalIncident.occurred_at.desc()).limit(limit).all()

        for inc, geojson_str in incidents:
            lat, lon = None, None
            if geojson_str:
                g = json.loads(geojson_str)
                coords = g.get("coordinates", [])
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
            formatted_list.append(FieldReportResponse(
                report_id=inc.incident_id,
                segment_id=inc.segment_id,
                reporter_name="Disaster Management Authority",
                reporter_role="Historical Ground-Truth",
                report_type=inc.incident_type,
                severity=inc.severity,
                status="VERIFIED",
                description=inc.description,
                photo_url=None,
                latitude=lat,
                longitude=lon,
                created_at=str(inc.occurred_at),
                updated_at=str(inc.created_at),
            ))
        total_count = len(formatted_list)

    return FieldReportListResponse(
        total_count=total_count,
        reports=formatted_list
    )


@router.get("/{report_id}", response_model=FieldReportResponse)
def get_field_report(
    report_id: int,
    db: Session = Depends(get_db),
):
    """
    Get detailed field report by report_id.
    """
    result = db.query(FieldReport, ST_AsGeoJSON(FieldReport.geom).label("geojson"))\
               .filter(FieldReport.report_id == report_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"Field report with id {report_id} not found")
    
    report, geojson_str = result
    return format_report_response(report, geojson_str)


@router.patch("/{report_id}/status", response_model=FieldReportResponse)
def update_field_report_status(
    report_id: int,
    status_update: FieldReportStatusUpdate,
    db: Session = Depends(get_db),
):
    """
    Update field report status in dashboard workflow (SUBMITTED, UNDER_REVIEW, VERIFIED, RESOLVED).
    """
    report = db.query(FieldReport).filter(FieldReport.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail=f"Field report with id {report_id} not found")

    new_status = status_update.status.upper().strip()
    if new_status not in ("SUBMITTED", "UNDER_REVIEW", "VERIFIED", "RESOLVED"):
        raise HTTPException(status_code=400, detail="Invalid status. Must be SUBMITTED, UNDER_REVIEW, VERIFIED, or RESOLVED")

    report.status = new_status
    db.commit()
    db.refresh(report)

    return format_report_response(report)
