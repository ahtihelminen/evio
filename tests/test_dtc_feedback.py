"""Feedback: a stage may read the chain's own outputs of the last step,
for the stage kinds its manifest names; nothing is fed to a stage that did
not ask, and nothing has to be used."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import replace

import pytest
from fakes.dtc_timer import FakeTimer
from fakes.dtc_toys import LOG, registered

from evio.dtc.chain import build
from evio.dtc.errors import ConfigError
from evio.dtc.facts import EXTENT
from evio.dtc.manifest import Manifest, Stage
from evio.dtc.result import Pipeline
from evio.dtc.types import NO_FEEDBACK, Output, Verdicts
from tests.support.dtc_chain import STEP_US, STREAMS, spec_of, step_at

pytestmark = pytest.mark.t1

CHAIN = ("toy-detector", "toy-feedback-tracker", "toy-classifier-every-step",
         "toy-decision")


@pytest.fixture(autouse=True)
def toys() -> Iterator[None]:
    with registered():
        yield


def make(*methods: str) -> Pipeline:
    return build(
        spec_of(*methods), {EXTENT: (1280, 720)}, streams=STREAMS,
        step_us=STEP_US, timer=FakeTimer(),
    )


def drive(pipeline: Pipeline, count: int) -> list:
    return [pipeline.step(step_at(i, *STREAMS)) for i in range(count)]


def fed(stream: str, index: int) -> Output:
    return next(
        f for s, i, f in LOG["feedback"] if (s, i) == (stream, index)
    )


def test_a_stage_that_asked_for_nothing_gets_the_empty_feedback() -> None:
    drive(make("toy-detector", "toy-tracker"), 2)

    assert all(f is NO_FEEDBACK for _, _, f in LOG["feedback"])


def test_a_tracker_reads_its_stream_s_last_verdicts_and_alarms() -> None:
    results = drive(make(*CHAIN), 2)

    assert fed("cam0", 0) == NO_FEEDBACK
    last = fed("cam0", 1)
    assert last.verdicts.items == tuple(
        v for v in results[0].verdicts if v.stream == "cam0"
    )
    assert last.alarms.items == tuple(
        a for a in results[0].alarms if a.stream == "cam0"
    )
    assert last.candidates.items == last.tracks.items == ()


def test_a_joint_tracker_reads_every_stream_s_last_verdicts() -> None:
    results = drive(
        make("toy-detector", "toy-joint-feedback-tracker",
             "toy-classifier-every-step"),
        2,
    )

    assert fed(None, 1).verdicts.items == results[0].verdicts


def test_a_classifier_reads_its_own_last_verdicts_aged_between_runs() -> None:
    drive(make("toy-detector", "toy-tracker", "toy-feedback-classifier"), 3)

    first_run = LOG["classifier-feedback"][:2]
    second_run = LOG["classifier-feedback"][2:]
    assert all(not v.items for v in first_run)
    assert {v.age_steps for f in second_run for v in f.items} == {1}


def test_a_batch_classifier_gets_the_feedback_on_each_call() -> None:
    drive(make("toy-detector", "toy-tracker",
               "toy-feedback-batch-classifier"), 2)

    assert [len(v.items) for v in LOG["batch-feedback"]] == [0, 0, 1, 1]


def test_feedback_from_a_stage_the_chain_lacks_is_empty() -> None:
    drive(make("toy-detector", "toy-feedback-tracker"), 2)

    assert fed("cam1", 1).verdicts == Verdicts()


def test_reset_forgets_the_last_step() -> None:
    pipeline = make(*CHAIN)
    drive(pipeline, 2)
    pipeline.reset()
    LOG["feedback"].clear()

    pipeline.step(step_at(0, *STREAMS))

    assert fed("cam0", 0) == NO_FEEDBACK


def test_a_source_may_not_set_feedback() -> None:
    forged = replace(step_at(0, *STREAMS), feedback=Output(verdicts=Verdicts()))

    with pytest.raises(ConfigError, match="feedback"):
        make(*CHAIN).step(forged)


def test_an_unknown_feedback_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="bad"):
        Manifest("bad", "1", (Stage.TRACK,), feedback=(9,))


def test_a_step_that_crossed_a_process_is_accepted() -> None:
    step = pickle.loads(pickle.dumps(step_at(0, *STREAMS)))

    make(*CHAIN).step(step)


def test_a_joint_tracker_s_regions_come_back_once() -> None:
    results = drive(
        make("toy-detector", "toy-joint-track-feedback-tracker"), 2,
    )

    assert len(fed(None, 1).tracks.priors) == len(results[0].tracks)


def test_a_joint_stage_keeps_the_last_step_of_a_stream_now_missing() -> None:
    pipeline = make("toy-detector", "toy-joint-feedback-tracker",
                    "toy-classifier-every-step")
    first = pipeline.step(step_at(0, *STREAMS))

    pipeline.step(step_at(1, *STREAMS, streams=STREAMS[:1]))

    assert fed(None, 1).verdicts.items == first.verdicts
