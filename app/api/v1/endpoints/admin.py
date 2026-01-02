"""
Admin API Endpoints - Super admin operations.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.dependencies import verify_super_admin
from app.models.organization import Organization
from app.repositories.organization_repository import OrganizationRepository
from app.repositories.anpr_repository import AnprDetectionRepository
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class OrganizationStatsResponse(BaseModel):
    """Organization with statistics."""
    id: int
    name: str
    code: str
    is_super_admin: bool
    is_active: bool
    camera_count: int
    detection_count: int
    created_at: datetime
    updated_at: datetime


class AllOrganizationsResponse(BaseModel):
    """Response with all organizations and their stats."""
    total_organizations: int
    organizations: List[OrganizationStatsResponse]


@router.get(
    "/organizations",
    response_model=AllOrganizationsResponse,
    summary="Get all organizations (Super Admin)",
    description="Get list of all organizations with statistics. Requires super admin access."
)
async def get_all_organizations(
    skip: int = 0,
    limit: int = 100,
    organization: Organization = Depends(verify_super_admin),
    db: Session = Depends(get_db)
):
    """
    Get all organizations with statistics.

    Requires:
    - X-API-Token header with super admin token

    Query parameters:
    - skip: Number of records to skip (pagination)
    - limit: Maximum records to return (max 100)

    Returns list of all organizations with camera and detection counts.
    """
    if limit > 100:
        limit = 100

    org_repo = OrganizationRepository(db)
    organizations = org_repo.get_all(skip=skip, limit=limit)

    org_stats_list = []
    for org in organizations:
        stats = org_repo.get_with_stats(org.id)
        org_stats_list.append(
            OrganizationStatsResponse(
                id=org.id,
                name=org.name,
                code=org.code,
                is_super_admin=org.is_super_admin,
                is_active=org.is_active,
                camera_count=stats["camera_count"],
                detection_count=stats["detection_count"],
                created_at=org.created_at,
                updated_at=org.updated_at
            )
        )

    total = org_repo.count()

    return AllOrganizationsResponse(
        total_organizations=total,
        organizations=org_stats_list
    )


@router.get(
    "/organizations/{org_id}/stats",
    response_model=OrganizationStatsResponse,
    summary="Get organization statistics (Super Admin)",
    description="Get detailed statistics for a specific organization. Requires super admin access."
)
async def get_organization_stats(
    org_id: int,
    organization: Organization = Depends(verify_super_admin),
    db: Session = Depends(get_db)
):
    """
    Get statistics for a specific organization.

    Requires:
    - X-API-Token header with super admin token

    Returns organization details with statistics.
    """
    org_repo = OrganizationRepository(db)
    org = org_repo.get_by_id(org_id)

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found"
        )

    stats = org_repo.get_with_stats(org_id)

    return OrganizationStatsResponse(
        id=org.id,
        name=org.name,
        code=org.code,
        is_super_admin=org.is_super_admin,
        is_active=org.is_active,
        camera_count=stats["camera_count"],
        detection_count=stats["detection_count"],
        created_at=org.created_at,
        updated_at=org.updated_at
    )


class SystemStatsResponse(BaseModel):
    """System-wide statistics."""
    total_organizations: int
    active_organizations: int
    total_detections: int
    pending_detections: int
    processing_detections: int
    success_detections: int
    failed_detections: int
    total_cameras: int


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics (Super Admin)",
    description="Get system-wide statistics across all organizations. Requires super admin access."
)
async def get_system_stats(
    organization: Organization = Depends(verify_super_admin),
    db: Session = Depends(get_db)
):
    """
    Get system-wide statistics.

    Requires:
    - X-API-Token header with super admin token

    Returns aggregated statistics across all organizations.
    """
    org_repo = OrganizationRepository(db)
    detection_repo = AnprDetectionRepository(db)

    # Get organization counts
    total_orgs = org_repo.count()
    active_orgs = len(org_repo.get_active())

    # Get detection statistics
    from app.models.anpr_detection import ProcessingStatus
    total_detections = detection_repo.count()
    pending = len(detection_repo.get_by_status(ProcessingStatus.PENDING))
    processing = len(detection_repo.get_by_status(ProcessingStatus.PROCESSING))
    success = len(detection_repo.get_by_status(ProcessingStatus.SUCCESS))
    failed = len(detection_repo.get_by_status(ProcessingStatus.FAILED))

    # Get total unique cameras across all orgs
    from sqlalchemy import func
    from app.models.anpr_detection import AnprDetection
    total_cameras = db.query(
        func.count(func.distinct(AnprDetection.camera_id))
    ).filter(
        AnprDetection.camera_id.isnot(None)
    ).scalar() or 0

    return SystemStatsResponse(
        total_organizations=total_orgs,
        active_organizations=active_orgs,
        total_detections=total_detections,
        pending_detections=pending,
        processing_detections=processing,
        success_detections=success,
        failed_detections=failed,
        total_cameras=total_cameras
    )
