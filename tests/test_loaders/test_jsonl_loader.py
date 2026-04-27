"""Tests for faithcheck.loaders.jsonl_loader."""
import json
from pathlib import Path

import pytest

from faithcheck.loaders.jsonl_loader import JsonlLoader, StreamingJsonlLoader
from faithcheck.models import TaskItem


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    items = [
        {
            "item_id": "math-001",
            "prompt": "What is 3 + 4?",
            "reference_cot": [
                {"index": 0, "text": "We add 3 and 4."},
                {"index": 1, "text": "3 + 4 = 7."},
            ],
            "ground_truth": "7",
        },
        {
            "item_id": "math-002",
            "prompt": "What is 10 - 3?",
            "reference_cot": [
                {"index": 0, "text": "Subtract 3 from 10."},
                {"index": 1, "text": "10 - 3 = 7."},
            ],
            "ground_truth": "7",
        },
    ]
    path = tmp_path / "tasks.jsonl"
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return path


class TestJsonlLoader:
    def test_load_all(self, sample_jsonl: Path):
        items = JsonlLoader.load(sample_jsonl)
        assert len(items) == 2
        assert all(isinstance(i, TaskItem) for i in items)

    def test_first_item_fields(self, sample_jsonl: Path):
        items = JsonlLoader.load(sample_jsonl)
        assert items[0].item_id == "math-001"
        assert items[0].prompt == "What is 3 + 4?"
        assert len(items[0].reference_cot) == 2

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            JsonlLoader.load(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_line(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(ValueError, match="line 1"):
            JsonlLoader.load(path)

    def test_missing_required_field(self, tmp_path: Path):
        path = tmp_path / "missing.jsonl"
        path.write_text(json.dumps({"item_id": "x", "prompt": "p"}) + "\n")
        with pytest.raises(ValueError):
            JsonlLoader.load(path)

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        items = JsonlLoader.load(path)
        assert items == []

    def test_hash_computation(self, sample_jsonl: Path):
        h1 = JsonlLoader.compute_hash(sample_jsonl)
        h2 = JsonlLoader.compute_hash(sample_jsonl)
        assert h1 == h2
        assert len(h1) == 64


class TestStreamingJsonlLoader:
    def test_stream_items(self, sample_jsonl: Path):
        items = list(StreamingJsonlLoader.stream(sample_jsonl))
        assert len(items) == 2

    def test_stream_partial_consumption(self, tmp_path: Path):
        path = tmp_path / "big.jsonl"
        with open(path, "w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "item_id": f"item-{i}",
                    "prompt": f"prompt {i}",
                    "reference_cot": [{"index": 0, "text": f"step {i}"}],
                    "ground_truth": str(i),
                }) + "\n")
        count = 0
        for _ in StreamingJsonlLoader.stream(path):
            count += 1
            if count == 5:
                break
        assert count == 5
