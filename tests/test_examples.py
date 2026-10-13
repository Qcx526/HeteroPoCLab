"""Committed examples must round-trip and remain independent of process hash order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from heteropoclab.domain import (
    ExecutionRecord,
    ExecutionRequest,
    InputEvent,
    Observation,
    PoCCandidate,
    RawEvidence,
    TriggerSpec,
    VerificationResult,
    VulnerabilitySpec,
)
from heteropoclab.platforms import CompatibilityDecision, PlatformCapability

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"


@pytest.mark.parametrize("hash_seed", ["0", "711"])
def test_example_generation_is_reproducible(tmp_path: Path, hash_seed: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "m0a" / "generate_json.py"),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    assert "Generated m0a-mock-flow.json" in completed.stdout
    for filename in ("m0a-mock-flow.json", "m0a-c8000-capability.json"):
        assert (tmp_path / filename).read_bytes() == (EXAMPLES / filename).read_bytes()


def test_full_flow_json_round_trip() -> None:
    payload = json.loads((EXAMPLES / "m0a-mock-flow.json").read_text(encoding="utf-8"))
    shapes = (
        ("platform_capability", PlatformCapability),
        ("vulnerability_spec", VulnerabilitySpec),
        ("trigger_spec", TriggerSpec),
        ("poc_candidate", PoCCandidate),
        ("execution_request", ExecutionRequest),
        ("input_event", InputEvent),
        ("execution_record", ExecutionRecord),
        ("verification_result", VerificationResult),
    )
    for key, model in shapes:
        item = model.model_validate_json(json.dumps(payload[key]))
        assert model.model_validate_json(item.to_json()) == item
    for raw in payload["raw_evidence"]:
        assert RawEvidence.model_validate_json(json.dumps(raw)).content
    for observation in payload["observations"]:
        assert Observation.model_validate_json(json.dumps(observation)).evidence_ids
    assert payload["verification_result"]["verdict"] == "NOT_VERIFIED"
    assert payload["verification_result"]["scope"] == "MOCK"


def test_c8000_example_propagates_unknown_without_executing() -> None:
    payload = json.loads((EXAMPLES / "m0a-c8000-capability.json").read_text(encoding="utf-8"))
    capability = PlatformCapability.model_validate_json(json.dumps(payload["platform_capability"]))
    decision = CompatibilityDecision.model_validate_json(
        json.dumps(payload["compatibility_decision"])
    )
    assert capability.isa.value == "PowerPC"
    assert capability.input_interfaces.value == ("UART", "CAN", "SPI", "GPIO")
    assert capability.debug_capabilities.value == ("vendor.non-GDB.JTAG",)
    assert decision.status.value == "UNKNOWN"
    assert not payload["can_execute"]
    assert {issue.dimension for issue in decision.issues} == {
        "isa_profile",
        "endian",
        "bit_width",
        "abi",
    }
