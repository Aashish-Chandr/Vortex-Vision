"""
Auth endpoints: token generation for dev/testing.
In production, integrate with your IdP (Cognito, Auth0, etc.)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.auth import create_access_token

router = APIRouter()

# Dev credentials — replace with DB-backed user store in production
# nosec B105 — these are intentional dev-only defaults, not real secrets
_DEV_USERS = {"admin": "vortex-admin-pass", "viewer": "vortex-viewer-pass"}  # nosec B105


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def login(req: LoginRequest):
    expected = _DEV_USERS.get(req.username)
    if not expected or expected != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(subject=req.username)
    return TokenResponse(access_token=token)
