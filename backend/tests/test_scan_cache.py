"""Tests for ScanCache — LRU cache keyed by (model_id, mode, prompt)."""

from neural_mri.core.scan_cache import ScanCache


def test_put_and_get_returns_cached():
    c = ScanCache(max_entries=5)
    c.put("gpt2", "T1", "hello", {"data": 1})
    assert c.get("gpt2", "T1", "hello") == {"data": 1}


def test_get_miss_returns_none():
    c = ScanCache()
    assert c.get("gpt2", "T1", "nonexistent") is None


def test_lru_eviction_at_max():
    c = ScanCache(max_entries=2)
    c.put("gpt2", "T1", "a", {"v": 1})
    c.put("gpt2", "T2", "b", {"v": 2})
    c.put("gpt2", "fMRI", "c", {"v": 3})  # should evict "a"
    assert c.get("gpt2", "T1", "a") is None
    assert c.get("gpt2", "T2", "b") == {"v": 2}
    assert c.get("gpt2", "fMRI", "c") == {"v": 3}


def test_lru_access_refreshes_order():
    c = ScanCache(max_entries=2)
    c.put("gpt2", "T1", "a", {"v": 1})
    c.put("gpt2", "T2", "b", {"v": 2})
    c.get("gpt2", "T1", "a")  # refresh "a"
    c.put("gpt2", "fMRI", "c", {"v": 3})  # should evict "b" not "a"
    assert c.get("gpt2", "T1", "a") == {"v": 1}
    assert c.get("gpt2", "T2", "b") is None


def test_same_prompt_different_mode_no_collision():
    c = ScanCache()
    c.put("gpt2", "T1", "hello", {"mode": "T1"})
    c.put("gpt2", "T2", "hello", {"mode": "T2"})
    assert c.get("gpt2", "T1", "hello") == {"mode": "T1"}
    assert c.get("gpt2", "T2", "hello") == {"mode": "T2"}


def test_invalidate_model_removes_only_target():
    c = ScanCache()
    c.put("gpt2", "T1", "a", {"m": "gpt2"})
    c.put("pythia", "T1", "a", {"m": "pythia"})
    c.invalidate_model("gpt2")
    assert c.get("gpt2", "T1", "a") is None
    assert c.get("pythia", "T1", "a") == {"m": "pythia"}


def test_invalidate_model_noop_if_no_match():
    c = ScanCache()
    c.put("gpt2", "T1", "a", {"v": 1})
    c.invalidate_model("unknown")
    assert c.get("gpt2", "T1", "a") == {"v": 1}


def test_clear_empties_all():
    c = ScanCache()
    c.put("gpt2", "T1", "a", {"v": 1})
    c.put("gpt2", "T2", "b", {"v": 2})
    c.clear()
    assert c.get("gpt2", "T1", "a") is None
    assert c.get("gpt2", "T2", "b") is None


def test_key_uses_md5_hash():
    key = ScanCache._key("gpt2", "T1", "hello")
    assert key.startswith("gpt2::T1::")
    assert len(key.split("::")[-1]) == 12  # md5[:12]


def test_put_overwrites_same_key():
    c = ScanCache()
    c.put("gpt2", "T1", "a", {"v": 1})
    c.put("gpt2", "T1", "a", {"v": 2})
    assert c.get("gpt2", "T1", "a") == {"v": 2}
