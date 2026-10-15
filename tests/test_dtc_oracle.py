"""Oracle and replay stages: ground truth as data, and conformance."""

from __future__ import annotations

import pytest

from evio.dtc.conformance import check_method
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Stage
from evio.dtc.methods import oracle
from evio.dtc.protocols import Context
from evio.dtc.registry import METHODS, create
from evio.dtc.types import Candidate, Candidates, Region, Step
from tests.support.dtc import empty_events, step_of, window

pytestmark = pytest.mark.t1

STEP_US = 10_000
CONTEXT = Context("cam", ("cam",), {}, STEP_US)
LABELS = {
    0: [[1, 2, 11, 12, "drone"], [30, 30, 40, 40, "bird"]],
    "2": [[5, 5, 9, 9, "drone"]],
}


def _step(index: int) -> Step:
    batch = empty_events()
    return step_of(index, STEP_US, window("cam", batch, 0, STEP_US))


def test_oracle_detect_emits_the_step_boxes_as_unit_score_candidates() -> None:
    method = create("oracle-detect", {"labels": LABELS}, CONTEXT)

    candidates = method.step(_step(0), ())

    assert candidates == Candidates(
        (
            Candidate("cam", Region(box=(1.0, 2.0, 11.0, 12.0)), 1.0),
            Candidate("cam", Region(box=(30.0, 30.0, 40.0, 40.0)), 1.0),
        )
    )


def test_unknown_step_has_no_boxes() -> None:
    method = create("oracle-detect", {"labels": LABELS}, CONTEXT)

    assert method.step(_step(1), ()) == Candidates()


def test_string_step_keys_from_a_spec_are_accepted() -> None:
    method = create("oracle-detect", {"labels": LABELS}, CONTEXT)

    (candidate,) = method.step(_step(2), ()).items

    assert candidate.region.box == (5.0, 5.0, 9.0, 9.0)


def test_config_is_normalised_to_immutable_tuples() -> None:
    config = oracle.OracleConfig(LABELS)

    assert config.labels[0] == (
        (1.0, 2.0, 11.0, 12.0, "drone"),
        (30.0, 30.0, 40.0, 40.0, "bird"),
    )
    with pytest.raises(TypeError):
        config.labels[3] = ()  # type: ignore[index]


def test_config_does_not_alias_the_input() -> None:
    raw = {0: [[1, 2, 3, 4, "a"]]}
    config = oracle.OracleConfig(raw)

    raw[0].append([9, 9, 9, 9, "b"])

    assert len(config.labels[0]) == 1


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        create("oracle-detect", {"boxes": {}}, CONTEXT)


def test_oracle_track_emits_labelled_tracks_and_live_ids() -> None:
    method = create("oracle-track", {"labels": LABELS}, CONTEXT)

    tracks = method.step(_step(0), Candidates())

    assert [(t.id, t.region.box, t.confidence) for t in tracks.items] == [
        ("drone", (1.0, 2.0, 11.0, 12.0), 1.0),
        ("bird", (30.0, 30.0, 40.0, 40.0), 1.0),
    ]
    assert tracks.live_ids == {"drone", "bird"}
    assert tracks.priors == ()


def test_oracle_track_ignores_the_candidates_it_is_given() -> None:
    method = create("oracle-track", {"labels": LABELS}, CONTEXT)
    noise = Candidates((Candidate("cam", Region(point=(0.0, 0.0)), 5.0),))

    assert method.step(_step(1), noise) == method.step(_step(1), Candidates())


def test_replay_detect_re_emits_recorded_boxes() -> None:
    method = create("replay-detect", {"recorded": LABELS}, CONTEXT)

    candidates = method.step(_step(2), ())

    assert [c.region.box for c in candidates.items] == [(5.0, 5.0, 9.0, 9.0)]
    assert method.step(_step(7), ()) == Candidates()


def test_replay_takes_recorded_not_labels() -> None:
    with pytest.raises(ConfigError):
        create("replay-detect", {"labels": LABELS}, CONTEXT)


def test_manifests_declare_the_stages() -> None:
    assert METHODS["oracle-detect"].manifest.stages == (Stage.DETECT,)
    assert METHODS["replay-detect"].manifest.stages == (Stage.DETECT,)
    assert METHODS["oracle-track"].manifest.stages == (Stage.TRACK,)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("oracle-detect", "labels"),
        ("oracle-track", "labels"),
        ("replay-detect", "recorded"),
    ],
)
def test_conformant_with_and_without_labels(name: str, key: str) -> None:
    facts = {EXTENT: (128, 96)}

    assert check_method(name, facts=facts) == []
    assert check_method(name, facts=facts, config={key: LABELS}) == []
