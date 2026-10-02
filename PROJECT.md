# evio

<!-- project-init:meta
created: 2026-10-02
linear_team: SN
linear_project: (none yet)
repo: git@git.sensofusion.com:ops/evio.git
integration_branch: dev
-->

## Scope

**evio is the base event-camera library.** It does everything that works on a laptop with or without a camera plugged in: read and write event data, open cameras, project pixels to the sky, render frames, and provide the building blocks of the detection chain.

There is no finish line. The milestones set maturity bars.

## Non-goals

Explicitly out of scope. Each line is something a reader might reasonably assume is included.

- Appliance concerns: worker orchestration, web UI, SAPIENT egress, boot lifecycle. These belong to ops.
- Model training and research. These stay in evior. Only promoted methods land here.
- Driver sources. The IMX636 and U3V C++ drivers stay in their own repos. evio only links them.
- Camera-specific or deployment-specific wiring of the chain, and trained model weights.

## Milestones

### M1 — evio has a test suite we can trust

**Goal:** Any change to evio can be judged correct or broken by running the tests alone, on a laptop or on real cameras, so people and agents can iterate without someone checking by hand.

**Completion requirements:**

- [ ] **R1.1** — Behaviour is covered. Every package has automated tests that check what it does, not just that it imports. They run on a laptop with no camera attached.
- [ ] **R1.2** — Fast where it matters. There is a quick tier for the edit loop and a broader tier before merge. Each test belongs to exactly one tier, and this is enforced.
- [ ] **R1.3** — Deterministic and independent. Tests give the same result on every run, in any order, with no network and no leftover files. A failure means a real problem.
- [ ] **R1.4** — Faithful stand-ins. The simulated cameras and other test doubles are shown to behave like the real ones, so a passing laptop run means something.
- [ ] **R1.5** — A hardware suite on real cameras. Each supported camera type has automated tests against a real device. They cover recording and recovery from crashes and unplugs. They are kept separate from the laptop suite and skip cleanly when no camera is attached.
- [ ] **R1.6** — The hardware suite runs continuously. At least one camera of each supported type is permanently connected to a CI runner and the hardware tests run on it, so a hardware regression is caught by CI. Running them on an OPS unit is possible but not required.
- [ ] **R1.7** — Reproducible driver setup. Someone with a clean Linux machine can build and link the drivers by following the docs, which is what lets a new runner join the pool.

### M2 — evio is established as a well-maintained repo

**Goal:** A newcomer or a consumer can trust that evio is in good order and stays that way.

**Completion requirements:**

- [ ] **R2.1** — A gate nothing bypasses. Lint and tests run automatically on every proposed change. Work that fails them does not reach the integration branch.
- [ ] **R2.2** — Clean boundaries. evio stays independent of ops and evior, and this is guarded mechanically, not by convention.
- [ ] **R2.3** — Readable documentation. The README gets someone from clone to a working setup. Each package has a short note on what it owns and what it does not.
- [ ] **R2.4** — An orderly branch flow. Changes land on `dev` through reviewed merge requests, and a human promotes `dev` to `main`.

### M3 — Consumers depend on evio with confidence

**Goal:** ops and evior can rely on evio and upgrade it without surprises.

**Completion requirements:**

- [ ] **R3.1** — Versioned releases. Consumers use tagged releases only, and a published tag never changes.
- [ ] **R3.2** — Breakage is caught early. A change that would break a consumer is noticed before it is released, not after.
- [ ] **R3.3** — A routine promotion path. Moving a proven method from evior into evio is a documented, repeatable step. The method then lives in one place only.

### M4 — Detection building blocks are a trustworthy shared toolkit

**Goal:** The detection pieces are easy to find, reuse and trust, and honest about what they assume.

**Completion requirements:**

- [ ] **R4.1** — One discoverable home. Every available detection method can be found in one place, with a short description.
- [ ] **R4.2** — Explicit assumptions. Anything that assumes a particular sensor, such as IMX636, is stated, so nobody mistakes it for general.
- [ ] **R4.3** — Usable without the heavy parts. The core pieces work without GPUs, ML frameworks or hardware. Optional capabilities degrade cleanly when their extras are missing.
- [ ] **R4.4** — No duplicates. A given method does not live in both evio and evior.

## Decisions

Append-only, newest first.

- **2026-10-02** — The test suite, including the hardware suite and the CI runner it runs on, is its own first milestone (M1), ahead of the repo-maintenance, consumer and detection milestones — a trustworthy suite gates building CI and autonomous agentic dev loops. Hardware tests need only a connected camera, not an OPS unit.
- **2026-10-02** — evio is split out of ops as its own repo; PROJECT.md written with Scope, Non-goals and four milestones (M1–M4); Linear project deferred. Milestones are written at the conceptual level (what we want to see), leaving technical specifics to issues.
- **2026-10-02** — The detection chain is scoped as building blocks, not the full chain; camera- or deployment-specific wiring and trained weights are non-goals — the chain is generic over `EventsBatch`, but wiring like `CameraDetector` encodes how an OPS unit runs.
