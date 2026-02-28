"""In-memory session manager for real-time collaboration."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from neural_mri.schemas.collab import PARTICIPANT_COLORS

logger = logging.getLogger(__name__)


@dataclass
class ParticipantConn:
    """A connected participant with WebSocket reference."""

    id: str
    display_name: str
    role: str  # "host" | "viewer"
    color: str
    ws: WebSocket
    joined_at: float = field(default_factory=time.time)


@dataclass
class Session:
    """An active collaboration session."""

    session_id: str
    host_id: str
    created_at: float = field(default_factory=time.time)
    participants: dict[str, ParticipantConn] = field(default_factory=dict)
    scan_state: dict[str, Any] | None = None

    @property
    def host(self) -> ParticipantConn | None:
        return self.participants.get(self.host_id)

    @property
    def participant_count(self) -> int:
        return len(self.participants)


class SessionManager:
    """In-memory session store. Sessions are ephemeral."""

    def __init__(self, max_sessions: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions

    def create_session(self, host_name: str) -> tuple[str, str]:
        """Create a session. Returns (session_id, host_participant_id)."""
        if len(self._sessions) >= self._max_sessions:
            self._evict_stale()
        session_id = uuid.uuid4().hex[:8]
        host_id = uuid.uuid4().hex[:12]
        session = Session(session_id=session_id, host_id=host_id)
        self._sessions[session_id] = session
        logger.info("Session created: %s by %s", session_id, host_name)
        return session_id, host_id

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session deleted: %s", session_id)
            return True
        return False

    def add_participant(
        self,
        session_id: str,
        display_name: str,
        ws: WebSocket,
        participant_id: str | None = None,
        role: str = "viewer",
    ) -> ParticipantConn | None:
        """Add a participant to a session. Returns the connection or None."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        pid = participant_id or uuid.uuid4().hex[:12]
        color_idx = len(session.participants) % len(PARTICIPANT_COLORS)
        conn = ParticipantConn(
            id=pid,
            display_name=display_name,
            role=role,
            color=PARTICIPANT_COLORS[color_idx],
            ws=ws,
        )
        session.participants[pid] = conn
        return conn

    def remove_participant(self, session_id: str, participant_id: str) -> str:
        """Remove participant. Returns 'session_closed' | 'removed' | 'not_found'."""
        session = self._sessions.get(session_id)
        if not session or participant_id not in session.participants:
            return "not_found"

        del session.participants[participant_id]

        if participant_id == session.host_id:
            logger.info("Host left session %s, marking for closure", session_id)
            return "session_closed"

        if not session.participants:
            del self._sessions[session_id]

        return "removed"

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "host_name": (s.host.display_name if s.host else "Unknown"),
                "participant_count": s.participant_count,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s.created_at)),
            }
            for s in self._sessions.values()
        ]

    def _evict_stale(self) -> None:
        """Remove sessions with 0 participants."""
        stale = [sid for sid, s in self._sessions.items() if s.participant_count == 0]
        for sid in stale:
            del self._sessions[sid]
