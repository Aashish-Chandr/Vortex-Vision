"""
JWT + API Key authentication for VortexVision API.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

# In production, store hashed API keys in DB. This is a dev default.
_VALID_API_KEYS = {"vortex-dev-key-change-me"}


class TokenData(BaseModel):
    sub: str
    exp: Optional[datetime] = None


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.api_secret_key, algorithm=settings.jwt_algorithm)


def _verify_jwt(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, settings.api_secret_key, algorithms=[settings.jwt_algorithm])
        return TokenData(sub=payload["sub"])
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")


async def get_current_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """Accepts either Bearer JWT or X-API-Key header."""
    if api_key and api_key in _VALID_API_KEYS:
        return f"apikey:{api_key[:8]}..."

    if bearer:
        token_data = _verify_jwt(bearer.credentials)
        return token_data.sub

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Dependency alias for protected routes
RequireAuth = Depends(get_current_user)
