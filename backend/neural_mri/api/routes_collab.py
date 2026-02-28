"""REST endpoints for collaboration session management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from neural_mri.core.session_manager import SessionManager

router = APIRouter()


def get_session_manager() -> SessionManager:
    from neural_mri.main import session_manager

    return session_manager


@router.post("/create")
async def create_session(display_name: str = "Host") -> dict:
    sm = get_session_manager()
    session_id, host_id = sm.create_session(display_name)
    return {
        "session_id": session_id,
        "host_id": host_id,
        "join_url": f"?session={session_id}",
    }


@router.get("/list")
async def list_sessions() -> list:
    sm = get_session_manager()
    return sm.list_sessions()


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    sm = get_session_manager()
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "host_id": session.host_id,
        "host_name": (session.host.display_name if session.host else "Unknown"),
        "participant_count": session.participant_count,
        "participants": [
            {
                "id": p.id,
                "display_name": p.display_name,
                "role": p.role,
                "color": p.color,
            }
            for p in session.participants.values()
        ],
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    sm = get_session_manager()
    if not sm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
