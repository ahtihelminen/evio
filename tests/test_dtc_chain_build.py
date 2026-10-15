"""The chain builder rejects a bad spec before any data flows."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fakes.dtc_toys import LOG, registered

from evio.dtc.chain import build
from evio.dtc.errors import ConfigError, ResourceError
from evio.dtc.facts import EXTENT
from evio.dtc.result import Pipeline
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of

pytestmark = pytest.mark.t1

importlib.import_module("evio.dtc.methods")


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def build_with(
    *methods: str | tuple[str, dict], facts: dict | None = None
) -> Pipeline:
    return build(
        spec_of(*methods),
        {EXTENT: (64, 48)} if facts is None else facts,
        streams=STREAMS,
        step_us=STEP_US,
    )


def test_a_chain_without_streams_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="at least one stream"):
        build(spec_of("toy-detector"), {}, streams=[], step_us=STEP_US)


def test_unknown_method_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown method 'nope'"):
        build_with("nope")


def test_config_with_an_unknown_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        build_with(("toy-detector", {"typo": 1}))


@pytest.mark.parametrize(
    "methods",
    [
        (),
        ("toy-tracker",),
        ("toy-detector", "toy-classifier"),
        ("toy-detector", "toy-decision"),
        ("toy-detector", "toy-detector"),
        ("toy-detector", "toy-tracker", "toy-decision"),
        ("toy-detect-track", "toy-tracker"),
    ],
)
def test_stages_that_do_not_line_up_are_a_config_error(
    methods: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigError, match="DETECT, TRACK"):
        build_with(*methods)


def test_missing_source_facts_are_named_in_the_config_error() -> None:
    with pytest.raises(ConfigError, match=r"'rate-excess' needs.*extent"):
        build_with("rate-excess", facts={})


def test_missing_dependency_is_a_resource_error() -> None:
    with pytest.raises(ResourceError, match="no_such_module_xyz"):
        build_with("toy-needs-module")


def test_unknown_evidence_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="ghost-view"):
        build_with("toy-wants-ghost")


def test_nothing_is_built_when_a_later_stage_fails_the_checks() -> None:
    with pytest.raises(ConfigError):
        build_with("toy-detector", "toy-decision")
    assert LOG["built"] == []


def test_a_stream_method_is_built_once_per_stream() -> None:
    build_with("toy-detector", "toy-tracker")
    assert sorted(LOG["built"]) == [
        ("detector", "cam0"),
        ("detector", "cam1"),
        ("tracker", "cam0"),
        ("tracker", "cam1"),
    ]


def test_a_joint_method_is_built_once_without_a_stream() -> None:
    pipeline = build_with("joint")
    assert list(pipeline.stats) == ["joint"]


def test_each_evidence_provider_is_built_once_per_stream() -> None:
    build_with(
        "toy-detector",
        "toy-tracker",
        "toy-classifier",
        "toy-decision",
    )
    assert sorted(LOG["provider-built"]) == ["cam0", "cam1"]


def test_a_built_chain_has_stats_per_stage_and_a_provenance() -> None:
    pipeline = build_with("toy-detector", "toy-tracker", "toy-classifier")
    assert list(pipeline.stats) == [
        "toy-detector",
        "toy-tracker",
        "toy-classifier",
        "evidence:toy-count",
    ]
    assert len(pipeline.provenance) == 16
