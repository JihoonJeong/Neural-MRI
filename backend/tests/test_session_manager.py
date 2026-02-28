"""Tests for SessionManager — in-memory collaboration sessions."""

from unittest.mock import MagicMock

from neural_mri.core.session_manager import SessionManager


def _mock_ws():
    return MagicMock()


def test_create_session():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    assert len(sid) == 8
    assert len(host_id) == 12
    assert sm.get_session(sid) is not None


def test_get_nonexistent_session():
    sm = SessionManager()
    assert sm.get_session("nope") is None


def test_delete_session():
    sm = SessionManager()
    sid, _ = sm.create_session("Alice")
    assert sm.delete_session(sid)
    assert sm.get_session(sid) is None


def test_delete_nonexistent():
    sm = SessionManager()
    assert not sm.delete_session("nope")


def test_add_participant_host():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    p = sm.add_participant(sid, "Alice", _mock_ws(), participant_id=host_id, role="host")
    assert p is not None
    assert p.role == "host"
    assert p.id == host_id
    assert sm.get_session(sid).participant_count == 1


def test_add_participant_viewer():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    sm.add_participant(sid, "Alice", _mock_ws(), participant_id=host_id, role="host")
    p2 = sm.add_participant(sid, "Bob", _mock_ws())
    assert p2 is not None
    assert p2.role == "viewer"
    assert sm.get_session(sid).participant_count == 2


def test_add_participant_invalid_session():
    sm = SessionManager()
    assert sm.add_participant("nope", "Bob", _mock_ws()) is None


def test_remove_viewer():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    sm.add_participant(sid, "Alice", _mock_ws(), participant_id=host_id, role="host")
    viewer = sm.add_participant(sid, "Bob", _mock_ws())
    result = sm.remove_participant(sid, viewer.id)
    assert result == "removed"
    assert sm.get_session(sid).participant_count == 1


def test_host_leaves_session_closed():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    sm.add_participant(sid, "Alice", _mock_ws(), participant_id=host_id, role="host")
    result = sm.remove_participant(sid, host_id)
    assert result == "session_closed"
    # Session still exists until explicit delete
    assert sm.get_session(sid) is not None


def test_remove_nonexistent():
    sm = SessionManager()
    assert sm.remove_participant("nope", "nope") == "not_found"


def test_list_sessions():
    sm = SessionManager()
    sm.create_session("Alice")
    sm.create_session("Bob")
    sessions = sm.list_sessions()
    assert len(sessions) == 2
    assert all("session_id" in s for s in sessions)


def test_max_sessions_evicts_stale():
    sm = SessionManager(max_sessions=2)
    sm.create_session("A")
    sm.create_session("B")
    # Both are stale (no participants), creating a 3rd should evict them
    sm.create_session("C")
    assert len(sm.list_sessions()) <= 2


def test_participant_colors_cycle():
    sm = SessionManager()
    sid, _ = sm.create_session("Host")
    colors = set()
    for i in range(10):
        p = sm.add_participant(sid, f"P{i}", _mock_ws())
        colors.add(p.color)
    # Should cycle through 8 predefined colors
    assert len(colors) <= 8


def test_session_host_property():
    sm = SessionManager()
    sid, host_id = sm.create_session("Alice")
    session = sm.get_session(sid)
    # No participants yet, host is None
    assert session.host is None
    sm.add_participant(sid, "Alice", _mock_ws(), participant_id=host_id, role="host")
    assert session.host is not None
    assert session.host.display_name == "Alice"
