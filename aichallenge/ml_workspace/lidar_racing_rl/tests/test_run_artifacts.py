"""Run artifact path and serialization contract tests."""

from __future__ import annotations

import json

import pytest

from lidar_racing_rl.sac.run_artifacts import (
    RepositorySnapshot,
    append_jsonl,
    write_run_artifacts,
)


def test_write_run_artifacts_is_complete_and_refuses_overwrite(tmp_path) -> None:
    snapshot = RepositorySnapshot(
        root_commit="abc123",
        root_status="## branch\n M file",
        root_diff="diff --git a/file b/file",
        submodule_commits={"repos/f1": "def456"},
        submodule_status={"repos/f1": "## branch"},
        submodule_diffs={"repos/f1": "diff --git a/sub b/sub"},
    )
    write_run_artifacts(
        tmp_path,
        resolved_config_yaml="seed: 7\n",
        repository=snapshot,
        environment={"python": "test"},
        lineage={"actor_initialization": {"step": 150000}},
    )

    assert (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8") == "seed: 7\n"
    assert json.loads((tmp_path / "submodule_commits.txt").read_text(encoding="utf-8")) == {
        "repos/f1": "def456"
    }
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["lineage"]["actor_initialization"]["step"] == 150000
    with pytest.raises(FileExistsError, match="provenance"):
        write_run_artifacts(
            tmp_path,
            resolved_config_yaml="seed: 7\n",
            repository=snapshot,
        )


def test_metrics_jsonl_rejects_non_finite_values(tmp_path) -> None:
    output = tmp_path / "metrics.jsonl"
    append_jsonl(output, {"step": 1, "loss": 0.5})
    with pytest.raises(ValueError):
        append_jsonl(output, {"step": 2, "loss": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0]) == {
        "loss": 0.5,
        "step": 1,
    }
