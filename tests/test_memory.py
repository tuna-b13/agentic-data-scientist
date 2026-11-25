"""Tests for the memory agent."""
import os
import pytest
from agents.memory import JSONMemory


@pytest.fixture
def mem(tmp_path):
    return JSONMemory(str(tmp_path / "test_memory.json"))


def test_empty_memory_returns_none(mem):
    assert mem.get_dataset_record("fp_nonexistent") is None


def test_upsert_and_retrieve(mem):
    mem.upsert_dataset_record("fp_123", {"best_model": "RF", "score": 0.9})
    record = mem.get_dataset_record("fp_123")
    assert record is not None
    assert record["best_model"] == "RF"


def test_upsert_overwrites(mem):
    mem.upsert_dataset_record("fp_123", {"best_model": "RF"})
    mem.upsert_dataset_record("fp_123", {"best_model": "GB"})
    assert mem.get_dataset_record("fp_123")["best_model"] == "GB"


def test_persistence(tmp_path):
    path = str(tmp_path / "persist.json")
    m1 = JSONMemory(path)
    m1.upsert_dataset_record("fp_1", {"val": 42})

    m2 = JSONMemory(path)
    assert m2.get_dataset_record("fp_1")["val"] == 42


def test_add_note(mem):
    mem.add_note("test note")
    assert len(mem.data["notes"]) == 1
    assert mem.data["notes"][0]["msg"] == "test note"


def test_corrupted_file_recovers(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        f.write("{{{invalid json")
    mem = JSONMemory(path)
    assert mem.get_dataset_record("anything") is None
    assert os.path.exists(path + ".bak")
