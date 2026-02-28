"""Pydantic models for the collaboration feature."""

from __future__ import annotations

from pydantic import BaseModel

PARTICIPANT_COLORS = [
    "#ff6b6b",
    "#4ecdc4",
    "#45b7d1",
    "#f7dc6f",
    "#bb8fce",
    "#82e0aa",
    "#f0b27a",
    "#85c1e9",
]


class Participant(BaseModel):
    id: str
    display_name: str
    role: str  # "host" | "viewer"
    color: str


class SessionInfo(BaseModel):
    session_id: str
    host_name: str
    participant_count: int
    created_at: str


class CreateSessionResponse(BaseModel):
    session_id: str
    host_id: str
    join_url: str
