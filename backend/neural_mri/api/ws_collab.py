"""WebSocket endpoint for real-time collaboration."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from neural_mri.core.session_manager import (
    ParticipantConn,
    Session,
    SessionManager,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_session_manager() -> SessionManager:
    from neural_mri.main import session_manager

    return session_manager


async def _broadcast(
    session: Session,
    message: dict,
    exclude_id: str | None = None,
) -> None:
    """Send message to all participants except exclude_id."""
    disconnected: list[str] = []
    for pid, participant in session.participants.items():
        if pid == exclude_id:
            continue
        try:
            await participant.ws.send_json(message)
        except Exception:
            disconnected.append(pid)
    for pid in disconnected:
        session.participants.pop(pid, None)


def _participant_dict(p: ParticipantConn) -> dict:
    return {
        "id": p.id,
        "display_name": p.display_name,
        "role": p.role,
        "color": p.color,
    }


async def _handle_message(
    msg: dict,
    participant: ParticipantConn,
    session: Session,
    ws: WebSocket,
) -> bool:
    """Handle a single message. Returns False if client wants to leave."""
    msg_type = msg.get("type")

    if msg_type == "c2s_scan_state":
        if participant.role != "host":
            await ws.send_json(
                {
                    "type": "s2c_error",
                    "message": "Only host can update scan state",
                }
            )
            return True
        payload = msg.get("payload", {})
        session.scan_state = payload
        await _broadcast(
            session,
            {"type": "s2c_scan_state", "payload": payload},
            exclude_id=participant.id,
        )

    elif msg_type == "c2s_cursor":
        await _broadcast(
            session,
            {
                "type": "s2c_cursor",
                "participant_id": participant.id,
                "display_name": participant.display_name,
                "x": msg.get("x", 0),
                "y": msg.get("y", 0),
            },
            exclude_id=participant.id,
        )

    elif msg_type == "c2s_select_layer":
        await _broadcast(
            session,
            {
                "type": "s2c_select_layer",
                "participant_id": participant.id,
                "layer_id": msg.get("layer_id"),
            },
            exclude_id=participant.id,
        )

    elif msg_type == "c2s_select_token":
        await _broadcast(
            session,
            {
                "type": "s2c_select_token",
                "participant_id": participant.id,
                "token_idx": msg.get("token_idx", 0),
            },
            exclude_id=participant.id,
        )

    elif msg_type == "c2s_ping":
        await ws.send_json({"type": "s2c_pong"})

    elif msg_type == "c2s_leave":
        return False

    else:
        await ws.send_json(
            {
                "type": "s2c_error",
                "message": f"Unknown type: {msg_type}",
            }
        )

    return True


@router.websocket("/ws/collab/{session_id}")
async def websocket_collab(ws: WebSocket, session_id: str) -> None:
    sm = _get_session_manager()
    session = sm.get_session(session_id)
    if not session:
        await ws.close(code=4004, reason="Session not found")
        return

    await ws.accept()
    participant: ParticipantConn | None = None

    try:
        # Wait for c2s_join
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        msg = json.loads(raw)
        if msg.get("type") != "c2s_join":
            await ws.send_json(
                {
                    "type": "s2c_error",
                    "message": "First message must be c2s_join",
                }
            )
            await ws.close()
            return

        display_name = msg.get("display_name", "Anonymous")
        host_id = msg.get("host_id")

        # Determine role
        if host_id and host_id == session.host_id:
            participant = sm.add_participant(
                session_id,
                display_name,
                ws,
                participant_id=host_id,
                role="host",
            )
        else:
            participant = sm.add_participant(session_id, display_name, ws)

        if not participant:
            await ws.send_json(
                {
                    "type": "s2c_error",
                    "message": "Failed to join session",
                }
            )
            await ws.close()
            return

        # Confirm join
        await ws.send_json(
            {
                "type": "s2c_joined",
                "session_id": session_id,
                "participant_id": participant.id,
                "role": participant.role,
                "participants": [_participant_dict(p) for p in session.participants.values()],
                "scan_state": session.scan_state,
            }
        )

        # Notify others
        await _broadcast(
            session,
            {
                "type": "s2c_participant_joined",
                "participant": _participant_dict(participant),
            },
            exclude_id=participant.id,
        )

        logger.info(
            "Participant %s (%s) joined session %s as %s",
            participant.id,
            display_name,
            session_id,
            participant.role,
        )

        # Main message loop
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "s2c_error", "message": "Invalid JSON"})
                continue

            keep_going = await _handle_message(msg, participant, session, ws)
            if not keep_going:
                break

    except WebSocketDisconnect:
        logger.info(
            "Collab participant disconnected from session %s",
            session_id,
        )
    except TimeoutError:
        logger.warning("Collab join timeout for session %s", session_id)
        try:
            await ws.close(code=4008, reason="Join timeout")
        except Exception:
            pass
    except Exception as exc:
        logger.exception(
            "Collab WebSocket error in session %s: %s",
            session_id,
            exc,
        )
    finally:
        if participant:
            is_host = participant.role == "host"

            if is_host:
                # Notify viewers before removing
                await _broadcast(
                    session,
                    {
                        "type": "s2c_session_closed",
                        "reason": "Host disconnected",
                    },
                    exclude_id=participant.id,
                )

            result = sm.remove_participant(session_id, participant.id)

            if result == "session_closed":
                sm.delete_session(session_id)
            elif result == "removed":
                remaining = sm.get_session(session_id)
                if remaining:
                    await _broadcast(
                        remaining,
                        {
                            "type": "s2c_participant_left",
                            "participant_id": participant.id,
                        },
                    )
