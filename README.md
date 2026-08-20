# Research Evidence Agent

A deliberately small, explainable research agent for a technical-interview
exercise. It answers questions from a six-note synthetic corpus while keeping
model decisions, deterministic execution, and explicit evidence separate.

## Why this is agentic

The next action is not hard-coded in the runtime. A `DecisionModel` chooses a
`ToolAction` or `FinalAction` from the question and previous observations, so a
later choice can depend on an earlier tool result. If the sequence were
completely predictable, a deterministic workflow would be preferable.

## Architecture

```text
User question
     |
     v
DecisionModel
     |
     v
ToolAction / FinalAction
     |
     +--> ToolRegistry --> deterministic tool --> observation --+
     |                                                          |
     +------------------------- next decision <------------------+
     |
     +--> FinalAction --> provenance gate --> answer / reject
```

`ResearchAgent` owns this explicit, single-agent observe-decide-act loop. Model
code selects actions; ordinary Python validates, executes, records, and stops.

## Deterministic tools

- `search_notes` performs lexical search over the local Markdown corpus.
- `read_note` retrieves one allow-listed document ID.
- `calculate` evaluates a small safe arithmetic syntax without `eval`.
- `check_constraints` checks explicit metadata comparisons and reports violations.

## Reliability boundaries

- An explicit registry exposes only the four tools above.
- Model actions use strict JSON structures parsed into typed actions.
- Tool failures become observations, allowing a later recovery action.
- `max_steps` bounds every run.
- The final-answer gate requires cited source IDs to have been read successfully.
- Missing measurements can produce a source-grounded abstention instead of a guess.

The provenance gate verifies that cited evidence was read; it does not prove
that every natural-language claim is factually correct. Model-output repair,
semantic claim verification, and robust prompt-injection defenses remain outside
this prototype.

## Model options

- `ScriptedModel` supplies deterministic testing, evaluation, and offline fallback.
- `TransformersDecisionModel` uses the open-weight `Qwen/Qwen3.5-2B` by default.
- Inference runs inside the Colab runtime; no cloud inference API is used.
- The model only chooses structured actions. `ResearchAgent` still executes and
  validates those actions.
- `ScriptedModel` remains the deterministic fallback.

Core runtime dependencies are standard-library only. Optional Colab dependencies
for PyTorch and image processing are installed with `pip install -e ".[colab]"`.
Because Qwen3.5 support is newer than an honest stable minimum currently captures,
the notebook installs Transformers directly from the official Hugging Face `main`
branch before installing this repository. The normal and test dependency sets are
unchanged.

## Run tests

```bash
python -m pytest -q
```

## Run deterministic evaluation

```bash
python eval/evaluate.py
```

The ten scripted cases measure observable runtime behavior such as tool
dispatch, grounding, recovery, abstention, arithmetic, constraints, and bounded
execution. They do not measure Qwen intelligence or open-ended answer quality,
and they do not download or invoke a model.

For a minimal offline example, run `python demo.py`.

## Colab demo

[`demo_colab.ipynb`](demo_colab.ipynb) demonstrates the same runtime with the
optional Transformers decision model and clearly labeled scripted fallback.

## Scope

This is intentionally a roughly one-day prototype, not a production platform.
Deliberate non-goals include multi-agent orchestration, a vector database, a web
UI, production authentication, and automatic model-output repair.
