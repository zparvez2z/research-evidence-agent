# AGENTS.md

## Project purpose

This repository contains a deliberately compact **Research Evidence Agent** for
exploring clear, inspectable agentic AI design.

The goal is to demonstrate the core mechanics of an agent that can choose its
next semantic action while deterministic software controls execution, evidence,
and safety boundaries. This is a focused prototype, not a production platform.

Prefer understandable code over clever abstractions, feature count, or framework
sophistication. Important behavior and architectural decisions should remain easy
to inspect, test, and explain.

## Core design

The system should preserve a genuine observe-decide-act loop:

1. A user provides a goal or question.
2. The model receives the goal, available tool schemas, and prior observable tool outcomes.
3. The model chooses the **next action** rather than following a fully predetermined workflow.
4. If it chooses a tool, deterministic application code executes the registered tool call.
5. The tool result or error becomes a structured observation in agent state.
6. The model sees the updated observable state and decides again.
7. If the model proposes a final answer, deterministic validation checks the project's provenance rules.
8. The run stops on an accepted final answer, a rejected final proposal, or the configured step limit.

The important agentic property is that a later action can depend on information
returned by earlier actions.

## Architectural principles

Separate three concerns clearly.

### Probabilistic model decisions

Use the model only where semantic interpretation or adaptive next-action selection
is useful, such as:

- understanding the user's goal,
- deciding which available tool to call next,
- interpreting observations,
- deciding whether to gather more evidence or propose a final answer.

### Deterministic software

Use ordinary Python for operations that can be handled more reliably without an
LLM, including:

- tool lookup and execution,
- strict action parsing,
- arithmetic,
- explicit constraint checks,
- stopping rules,
- final-answer/evidence validation,
- error handling.

Potential future deterministic guards, such as repeated-action detection, should
be added only when their behavior is explicit and tested.

### Explicit knowledge and evidence

Answers should be grounded in the local evidence corpus. The system should track
observable tool calls and successfully read documents, and it should be able to
return a grounded insufficient-evidence answer when measurements are unavailable.

## Core tools

The agent exposes exactly these four tools unless the project scope is explicitly
changed:

1. `search_notes(query, limit=5)`
   - Search the small local Markdown corpus lexically.
   - Return document identifiers, titles, and short relevant snippets.
   - Search results are discovery information, not authoritative final evidence.

2. `read_note(document_id)`
   - Read one known document from the corpus.
   - Return authoritative note content and metadata.
   - Reject unknown document identifiers.

3. `calculate(expression)`
   - Perform simple arithmetic deterministically using an AST allowlist.
   - Never use unrestricted `eval` or arbitrary code execution.

4. `check_constraints(document_id, requirements)`
   - Load authoritative metadata by document ID.
   - Evaluate explicit numerical or boolean requirements deterministically.
   - Return structured checks, pass/fail status, and violations.

Do not add tools merely to make the project appear more sophisticated.

## Final-answer gate

The model does not have unconditional authority to finalize a run.

The implemented provenance gate currently enforces these rules:

- at least one `read_note` call must have completed successfully,
- a final action must contain at least one evidence ID,
- every cited evidence ID must correspond to a document successfully read during the current run.

The gate is intentionally narrow. It does **not** prove that every natural-language
claim is true and it does **not** currently enforce that applicable hard
constraints were checked before finalization.

A rejected final proposal ends the current run with `status="final_rejected"`.
Do not describe the current implementation as automatically feeding a rejection
back to the model and continuing unless that runtime behavior is explicitly added.

## Required safety and failure behavior

Keep these behaviors visible and testable:

- unknown tool name -> explicit error,
- malformed tool arguments -> explicit error,
- ordinary tool exception -> structured error observation so a later decision can recover,
- excessive model decisions -> deterministic `max_steps` stop,
- unsupported final answer -> rejection by the provenance gate,
- insufficient measurements -> grounded insufficient-evidence answer when supporting source material exists,
- malformed model action JSON -> strict parse failure rather than silent repair.

Prompt injection inside retrieved documents is an important security concern, but
robust production defenses are outside the current scope and should be documented
as a limitation or future improvement.

## Model abstraction and testability

Keep the agent runtime separate from the model/provider.

The provider-neutral interface is conceptually:

```python
class DecisionModel(Protocol):
    def decide(
        self,
        question: str,
        observations: Sequence[ToolObservation],
        tools: Sequence[ToolSpec],
    ) -> Action:
        ...
```

`ScriptedModel` supports deterministic tests and evaluation so the agent loop,
tool execution, provenance validation, and failure behavior can be exercised
independently of real-model quality or hardware availability.

`TransformersDecisionModel` is the optional real-model adapter. Provider/model
logic must not leak into `ResearchAgent` or the deterministic tools.

## State and observability

Keep agent state explicit and small. The run should be easy to inspect from
externally meaningful state.

Current observable state includes:

- original question,
- ordered tool observations,
- step number per tool observation,
- tool name and arguments,
- tool result or error,
- final run status and evidence IDs.

Do **not** request, expose, persist, or display private model chain-of-thought.
Logs and demos should show only externally meaningful action/output information.

## Model-output contract

The preferred Qwen-facing action shapes are:

```json
{"action":"search_notes","arguments":{"query":"..."}}
```

and:

```json
{"action":"final","answer":"...","evidence_ids":["document-id"]}
```

The parser also accepts the earlier `type/tool/arguments` and
`type/final/answer/evidence_ids` shapes for backward compatibility with scripted
evaluation cases.

Do not broaden parsing into heuristic repair of malformed output without an
explicit design decision and tests. Native model tool calling or schema-constrained
generation is preferable to silently correcting arbitrary malformed JSON.

## Evaluation

Keep a small transparent deterministic evaluation set covering multiple runtime
behaviors rather than only happy paths.

Current evaluation categories include:

- factual retrieval,
- local/API metadata facts,
- unique constraint matching,
- arithmetic comparison,
- GPU-memory facts,
- unavailable energy measurement,
- unavailable training-cost measurement,
- latency-comparability limitations,
- tool-error recovery,
- maximum-step safety.

Evaluation results describe deterministic runtime scenarios. They must not be
presented as open-ended Qwen accuracy or general model intelligence.

Do not invent successful evaluation numbers. Report actual results, including
failures.

## Scope constraints

Unless the scope is explicitly changed:

- Use **one agent**, not a multi-agent system.
- Keep the main loop in ordinary Python so behavior remains visible.
- Do **not** use LangChain.
- Do **not** use LangGraph.
- Do **not** use a vector database.
- Do **not** add a web frontend.
- Do **not** add a database without a concrete need.
- Do **not** add Docker, Kubernetes, cloud deployment machinery, or authentication by default.
- Do **not** add autonomous web browsing.
- Do **not** add unnecessary design patterns or abstraction layers.
- Keep dependencies minimal.
- Prefer standard-library solutions when they remain clear and safe.

The project is intentionally focused. A small, well-understood system with
explicit limitations is preferable to unexplained complexity.

## Python and code-quality rules

- Target Python 3.11+.
- Use type hints for public interfaces and functions.
- Prefer dataclasses, typed structures, or small validation objects only where they improve clarity.
- Keep functions and modules focused.
- Use clear names rather than clever generic abstractions.
- Use `pytest` for tests.
- Keep tests deterministic where possible.
- Raise or return meaningful errors rather than silently swallowing failures.
- Never commit API keys, tokens, `.env` secrets, or credentials.

## Repository shape

The implemented structure is centered on:

```text
research-evidence-agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── demo.py
├── real_demo.py
├── demo_colab.ipynb
├── data/
│   └── notes/
├── docs/
│   └── architecture/
├── src/
│   └── research_agent/
│       ├── actions.py
│       ├── agent.py
│       ├── model.py
│       ├── state.py
│       ├── transformers_model.py
│       ├── validation.py
│       └── tools/
│           ├── calculator.py
│           ├── constraints.py
│           ├── notes.py
│           └── registry.py
├── tests/
└── eval/
    ├── cases.json
    └── evaluate.py
```

Prefer the smallest structure justified by implemented behavior.

## Working rules

Before changing code:

1. Read this file fully.
2. Inspect the existing repository and relevant tests.
3. Identify the requested behavior and the files that need to change.
4. Do not broaden scope without a concrete reason.

After changing code:

1. Run the relevant tests and checks.
2. Report what changed.
3. Explain any dependency added and why it is necessary.
4. Call out unfinished parts and limitations explicitly.
5. Do not perform large unrelated refactors.

## Architecture decision rule

For every nontrivial implementation choice, ask:

> Why should the model decide this rather than deterministic code?

If deterministic code can perform the operation more reliably and transparently,
prefer deterministic code. Use model autonomy where the next action genuinely
depends on interpreting the user's goal or observations from previous steps.
