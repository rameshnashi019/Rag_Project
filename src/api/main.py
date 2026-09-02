"""FastAPI authentication and chatbot endpoints."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from project.pipeline import RAGPipeline

load_dotenv()
logger = logging.getLogger(__name__)
STATIC_DIRECTORY = Path(__file__).parent / "static"

app = FastAPI(title="PDF RAG API", version="0.1.0")
bearer_scheme = HTTPBearer(auto_error=False)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the PDF chatbot web interface."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


class LoginRequest(BaseModel):
    """Credentials submitted to the login endpoint."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ChatRequest(BaseModel):
    """User question for the chatbot."""

    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    """Generated chatbot response."""

    answer: str
    sources: list[dict[str, object]]


def _required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set in .env")
    return value


def _create_token(username: str) -> tuple[str, int]:
    secret = _required_setting("JWT_SECRET_KEY")
    expires_in = int(os.getenv("JWT_EXPIRE_MINUTES", "60")) * 60
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    token = jwt.encode(
        {"sub": username, "exp": expires_at},
        secret,
        algorithm="HS256",
    )
    return token, expires_in


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Validate the bearer token and return its username."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    try:
        payload = jwt.decode(
            credentials.credentials,
            _required_setting("JWT_SECRET_KEY"),
            algorithms=["HS256"],
        )
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise JWTError
        return username
    except (JWTError, RuntimeError):
        logger.warning("Rejected invalid authentication token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """Authenticate the configured user and return a JWT."""
    expected_username = _required_setting("AUTH_USERNAME")
    expected_password = _required_setting("AUTH_PASSWORD")
    if request.username != expected_username or request.password != expected_password:
        logger.warning("Failed login attempt for username=%s", request.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token, expires_in = _create_token(request.username)
    logger.info("Successful login for username=%s", request.username)
    return LoginResponse(access_token=token, expires_in=expires_in)


@lru_cache(maxsize=1)
def _get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, username: str = Depends(get_current_user)) -> ChatResponse:
    """Answer an authenticated question using indexed PDF content."""
    try:
        answer, sources = _get_pipeline().ask_with_sources(request.question, k=request.k)
        logger.info("Answered chat request for username=%s", username)
        return ChatResponse(answer=answer, sources=sources)
    except Exception:
        logger.exception("Chat request failed for username=%s", username)
        raise HTTPException(status_code=503, detail="Chat service is temporarily unavailable") from None


def run() -> None:
    """Start the API with Uvicorn."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )