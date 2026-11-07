from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_has_reported_checkpoint_labels():
    text = (ROOT / "configs" / "experiment_protocol.yaml").read_text(encoding="utf-8")
    assert "inference_checkpoint: 9999" in text
    assert text.count("inference_checkpoint: 4999") == 2


def test_release_contains_no_machine_specific_paths_or_hosts():
    banned = re.compile(
        "(?:" + "/" + "data" + "/|" + "/" + "home" + "/|" +
        "\\\\" + "Users" + "\\\\|" + "\\\\" + "user" + "\\\\|" +
        "h" + "100" + "-|" + "s" + "sh" + "\\s|" +
        "s" + "cp" + "\\s|" + "r" + "sync" + "\\s|" + "r" + "6lite|" +
        "quest" + "_local21|" + "cib" + "_)", re.I
    )
    allowed = {".gitignore", "check_release.py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in allowed or ".git" in path.parts or "tests" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".sh", ".yaml", ".yml", ".md", ".toml", ".env"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not banned.search(text), path


def test_release_contains_no_generated_python_artifacts():
    for path in ROOT.rglob("*"):
        assert path.name != "__pycache__", path
        assert path.suffix.lower() not in {".pyc", ".pyo"}, path


def test_release_has_one_readme_only():
    markdown = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.md"))
    assert markdown == ["README.md"]


def test_public_checkpoint_slots_are_logical_paths():
    text = (ROOT / "configs" / "checkpoints.example.yaml").read_text(encoding="utf-8")
    assert "M4_smoothed:" in text
    assert "M4_smoothed_off:" in text
    assert ("/" + "data" + "/") not in text


def test_m4_representation_and_mask_contract():
    import numpy as np

    from egoweave_pretrain.representation import (
        REPRESENTATION_VERSION,
        encode_many_masked,
        pack_pose,
    )

    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (6, 4, 4)).copy()
    landmarks = np.zeros((6, 2, 21, 3), dtype=np.float32)
    landmarks[:, :, :, 0] = np.arange(21, dtype=np.float32)
    pose = pack_pose(transforms, transforms, transforms, landmarks)
    packet = {
        "pose153": pose,
        "anchors": np.asarray([0, 1], dtype=np.int64),
        "camera_valid": np.ones(6, dtype=bool),
        "head_valid": np.ones(6, dtype=bool),
        "hand_valid": np.ones((6, 2), dtype=bool),
        "alignment_version": np.asarray(REPRESENTATION_VERSION),
    }
    packet["hand_valid"][1:, 1] = False
    states, actions, state_mask, action_mask = encode_many_masked(packet, packet["anchors"], horizon=2)
    assert states.shape == (2, 156)
    assert actions.shape == (2, 2, 153)
    assert state_mask.shape == states.shape
    assert action_mask.shape == actions.shape
    assert not state_mask[1, 81:153].any()
    assert np.all(actions[1, :, 81:153] == 0)


def test_m4_checkpoint_transfer_allows_only_action_interface():
    import numpy as np

    from egoweave_pretrain.checkpoint import transfer_base_to_153

    source = {
        "trunk": np.zeros((2, 2), dtype=np.float32),
        "action_in_proj/kernel": np.zeros((32, 4), dtype=np.float32),
        "action_out_proj/kernel": np.zeros((4, 32), dtype=np.float32),
        "action_out_proj/bias": np.zeros((32,), dtype=np.float32),
    }
    target = {
        "trunk": np.zeros((2, 2), dtype=np.float32),
        "action_in_proj/kernel": np.zeros((153, 4), dtype=np.float32),
        "action_out_proj/kernel": np.zeros((4, 153), dtype=np.float32),
        "action_out_proj/bias": np.zeros((153,), dtype=np.float32),
    }
    result, audit = transfer_base_to_153(source, target)
    assert set(result) == set(target)
    assert len(audit["random_interface"]) == 3
