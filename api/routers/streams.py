"""
Stream management endpoints: add/remove/list video streams.
Persists stream state to database.
"""
import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import RequireAuth
from api.database import StreamRecord, get_db

logger = logging.getLogger(__name__)
router = APIRouter()

_readers: dict = {}


class StreamRequest(BaseModel):
    stream_id: str
    source: str
    fps_limit: int = 30


class StreamInfo(BaseModel):
    stream_id: str
    source: str
    active: bool


@router.get("/", response_model=List[StreamInfo])
async def list_streams(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(StreamRecord))
    rows = result.scalars().all()
    return [StreamInfo(stream_id=r.stream_id, source=r.source, active=bool(r.active)) for r in rows]


@router.post("/", status_code=201, dependencies=[RequireAuth])
async def add_stream(req: StreamRequest, request: Request, db: AsyncSession = Depends(get_db)):
    from ingestion.stream_reader import StreamReader, StreamConfig

    existing = await db.execute(select(StreamRecord).where(StreamRecord.stream_id == req.stream_id))
    if existing.scalar_one_or_none():
        raise HTTPException(400, f"Stream '{req.stream_id}' already exists")

    record = StreamRecord(stream_id=req.stream_id, source=req.source, fps_limit=req.fps_limit, active=1)
    db.add(record)
    await db.commit()

    config = StreamConfig(source=req.source, stream_id=req.stream_id, fps_limit=req.fps_limit)
    reader = StreamReader(config)
    _readers[req.stream_id] = reader
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, reader.start)

    logger.info("Stream started: %s → %s", req.stream_id, req.source)
    return {"message": f"Stream '{req.stream_id}' started"}


@router.delete("/{stream_id}", dependencies=[RequireAuth])
async def remove_stream(stream_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(StreamRecord).where(StreamRecord.stream_id == stream_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(404, f"Stream '{stream_id}' not found")

    if stream_id in _readers:
        _readers[stream_id].stop()
        del _readers[stream_id]

    record.active = 0
    await db.commit()
    return {"message": f"Stream '{stream_id}' stopped"}
