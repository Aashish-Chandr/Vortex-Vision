"""
Natural language query endpoint.
POST /query  { "question": "Show me all red cars speeding in the last 5 minutes" }
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import RequireAuth
from api.database import QueryHistoryRecord, get_db

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    time_window_seconds: Optional[float] = 300.0
    stream_id: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    clips_found: int
    processing_ms: float


@router.post("/", response_model=QueryResponse, dependencies=[RequireAuth])
async def natural_language_query(
    req: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    state = getattr(request.app.state, "vv", None)
    if state is None or state.nl_engine is None:
        raise HTTPException(
            503, "VLM query engine not ready. Check VLM_MODE and VLM_API_BASE settings."
        )

    result = state.nl_engine.query(
        question=req.question,
        time_window_seconds=req.time_window_seconds,
    )

    record = QueryHistoryRecord(
        question=result.query,
        answer=result.answer,
        clips_found=len(result.clips),
        processing_ms=result.processing_ms,
    )
    db.add(record)
    await db.commit()

    return QueryResponse(
        question=result.query,
        answer=result.answer,
        clips_found=len(result.clips),
        processing_ms=result.processing_ms,
    )


@router.get("/history")
async def query_history(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(QueryHistoryRecord)
        .order_by(desc(QueryHistoryRecord.created_at))
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "question": r.question,
            "answer": r.answer,
            "clips_found": r.clips_found,
            "processing_ms": r.processing_ms,
        }
        for r in rows
    ]
