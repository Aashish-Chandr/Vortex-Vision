"""
Anomaly event endpoints: list, filter, and retrieve detected events.
"""
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AnomalyEventRecord, get_db

router = APIRouter()


class EventOut(BaseModel):
    id: int
    stream_id: str
    timestamp: float
    frame_id: Optional[int]
    anomaly_type: Optional[str]
    confidence: Optional[float]
    autoencoder_score: Optional[float]
    transformer_score: Optional[float]
    clip_path: Optional[str]
    description: Optional[str]

    class Config:
        from_attributes = True


@router.get("/", response_model=List[EventOut])
async def get_events(
    stream_id: Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None),
    since: Optional[float] = Query(None, description="Unix timestamp lower bound"),
    limit: int = Query(100, le=1000),
    db: AsyncSession = Depends(get_db),
):
    q = select(AnomalyEventRecord).order_by(desc(AnomalyEventRecord.timestamp))
    if stream_id:
        q = q.where(AnomalyEventRecord.stream_id == stream_id)
    if anomaly_type:
        q = q.where(AnomalyEventRecord.anomaly_type == anomaly_type)
    if since:
        q = q.where(AnomalyEventRecord.timestamp >= since)
    q = q.limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_result = await db.execute(select(func.count()).select_from(AnomalyEventRecord))
    total = total_result.scalar()

    type_result = await db.execute(
        select(AnomalyEventRecord.anomaly_type, func.count().label("count"))
        .group_by(AnomalyEventRecord.anomaly_type)
    )
    by_type = {row.anomaly_type: row.count for row in type_result}

    return {"total": total, "by_type": by_type, "timestamp": time.time()}
