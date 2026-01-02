"""
Repository for ANPR Detection data access.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.models.anpr_detection import AnprDetection, ProcessingStatus
from app.repositories.base_repository import BaseRepository


class AnprDetectionRepository(BaseRepository[AnprDetection]):
    """Repository for ANPR Detection database operations."""

    def __init__(self, db: Session):
        super().__init__(AnprDetection, db)

    def get_by_organization(
        self,
        organization_id: int,
        skip: int = 0,
        limit: int = 100,
        status: Optional[ProcessingStatus] = None
    ) -> List[AnprDetection]:
        """Get detections for a specific organization."""
        query = self.db.query(self.model).filter(
            self.model.organization_id == organization_id
        )

        if status:
            query = query.filter(self.model.status == status)

        return query.order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()

    def get_by_client_detection_id(
        self,
        organization_id: int,
        client_detection_id: str
    ) -> Optional[AnprDetection]:
        """Get detection by client's tracking ID."""
        return (
            self.db.query(self.model)
            .filter(
                self.model.organization_id == organization_id,
                self.model.client_detection_id == client_detection_id
            )
            .first()
        )

    def get_by_camera(
        self,
        organization_id: int,
        camera_id: str,
        skip: int = 0,
        limit: int = 100,
        status: Optional[ProcessingStatus] = None
    ) -> List[AnprDetection]:
        """Get detections for a specific camera."""
        query = self.db.query(self.model).filter(
            self.model.organization_id == organization_id,
            self.model.camera_id == camera_id
        )

        if status:
            query = query.filter(self.model.status == status)

        return query.order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()

    def get_by_status(
        self,
        status: ProcessingStatus,
        skip: int = 0,
        limit: int = 100
    ) -> List[AnprDetection]:
        """Get detections by processing status."""
        return (
            self.db.query(self.model)
            .filter(self.model.status == status)
            .order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_pending_for_retry(self, max_retries: int = 3) -> List[AnprDetection]:
        """Get failed detections that haven't exceeded max retries."""
        return (
            self.db.query(self.model)
            .filter(
                self.model.status.in_([ProcessingStatus.FAILED, ProcessingStatus.RETRYING]),
                self.model.retry_count < max_retries
            )
            .order_by(desc(self.model.created_at))
            .all()
        )

    def get_stats_by_organization(self, organization_id: int) -> Dict[str, Any]:
        """Get processing statistics for an organization."""
        total = self.db.query(func.count(self.model.id)).filter(
            self.model.organization_id == organization_id
        ).scalar()

        success = self.db.query(func.count(self.model.id)).filter(
            self.model.organization_id == organization_id,
            self.model.status == ProcessingStatus.SUCCESS
        ).scalar()

        failed = self.db.query(func.count(self.model.id)).filter(
            self.model.organization_id == organization_id,
            self.model.status == ProcessingStatus.FAILED
        ).scalar()

        pending = self.db.query(func.count(self.model.id)).filter(
            self.model.organization_id == organization_id,
            self.model.status == ProcessingStatus.PENDING
        ).scalar()

        processing = self.db.query(func.count(self.model.id)).filter(
            self.model.organization_id == organization_id,
            self.model.status == ProcessingStatus.PROCESSING
        ).scalar()

        return {
            "total": total or 0,
            "success": success or 0,
            "failed": failed or 0,
            "pending": pending or 0,
            "processing": processing or 0,
            "success_rate": round((success / total * 100), 2) if total > 0 else 0
        }

    def search_by_numberplate(
        self,
        numberplate: str,
        organization_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AnprDetection]:
        """Search detections by numberplate text."""
        query = self.db.query(self.model).filter(
            self.model.numberplate_text.ilike(f"%{numberplate}%")
        )

        if organization_id:
            query = query.filter(self.model.organization_id == organization_id)

        return query.order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()

    def get_recent(
        self,
        limit: int = 10,
        organization_id: Optional[int] = None
    ) -> List[AnprDetection]:
        """Get most recent detections."""
        query = self.db.query(self.model)

        if organization_id:
            query = query.filter(self.model.organization_id == organization_id)

        return query.order_by(desc(self.model.created_at)).limit(limit).all()

    def get_all_with_filters(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[ProcessingStatus] = None,
        camera_id: Optional[str] = None
    ) -> List[AnprDetection]:
        """Get all detections across all organizations with optional filters (super admin)."""
        query = self.db.query(self.model)

        if status:
            query = query.filter(self.model.status == status)

        if camera_id:
            query = query.filter(self.model.camera_id == camera_id)

        return query.order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()
