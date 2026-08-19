# AGENTS.md

## Project purpose

This repository contains a deliberately small **Research Evidence Agent** built for a technical interview exercise for an Agentic AI student-assistant role.

The goal is to demonstrate the core mechanics of an AI agent clearly and defensibly. This is **not** a production platform and should not grow into one.

The candidate must be able to explain every important line and every architectural decision during a live interview. Prefer understandable code over clever abstractions, feature count, or framework sophistication.

## Core design

The system should demonstrate a genuine observe-decide-act loop:

1. A user provides a goal/question.
2. The model receives the goal, available tool schemas, and relevant prior observations.
3. The model chooses the **next action** rather than following a fully predetermined workflow.
4. If it chooses a tool, deterministic application code validates and executes the tool call.
5. The tool result becomes a structured observation in agent state.
6. The model sees the updated state and decides again.
7. If the model proposes a final answer, deterministic validation decides whether the system has enough evidence to finalize.
8. The run stops on a valid final answer or a safety limit.

The important agentic property is that the next action can depend on the results of earlier actions.

## Architectural principles

Separate three concerns clearly:

### Probabilistic model decisions

Use the model only where semantic interpretation or adaptive next-action selection is useful, such as:

- understanding the user's goal,
- deciding which available tool to call next,
- interpreting observations,
- deciding whether to gather more evidence or propose a final answer.

### Deterministic software

Use ordinary Python for things that can be handled more reliably without an LLM, including:

- tool lookup and execution,
- argument/schema validation,
- arithmetic,
- explicit constraint checks,
- stopping rules,
- repeated-action detection,
- final-answer/evidence validation,
- error handling.

### Explicit knowledge/evidence

Answers should be grounded in the local evidence corpus. The system should track which documents were searched/read and should be able to abstain when the available evidence does not support an answer.

## Core tools

The initial agent may use only these four tools unless the user explicitly approves another tool:

1. `search_notes(query)`
   - Search the small local Markdown corpus.
   - Return document identifiers/titles and short relevant snippets or match information.

2. `read_note(document_id)`
   - Read one known document from the corpus.
   - Reject unknown or unsafe paths/identifiers.

3. `calculate(expression)`
   - Perform simple safe arithmetic deterministically.
   - Never use unrestricted `eval`.

4. `check_constraints(candidate, requirements)`
   - Evaluate explicit numerical/boolean requirements deterministically.
   - Return structured pass/fail results and violations.

Do not add tools merely to make the project look more impressive.

## Final-answer gate

Do not let the model have unconditional authority to terminate a run.

Before accepting a final answer, deterministic code should be able to enforce simple requirements such as:

- at least one relevant source was actually read when factual evidence is required,
- referenced/cited source identifiers exist in the run state,
- requested hard constraints were checked when applicable,
- the maximum-step limit has not been exceeded.

If a final proposal fails validation, feed a concise structured observation back to the agent so it can choose another action.

The exact final validation rules should stay small, explicit, and easy to explain.

## Required safety/failure behavior

Design the prototype so these cases are visible and testable:

- unknown tool name -> structured error, not a crash,
- malformed tool arguments -> validation error,
- tool exception -> structured observation so the agent can recover or stop,
- repeated identical actions -> detectable loop protection,
- excessive steps -> deterministic safe stop,
- premature final answer -> rejected by the final-answer gate,
- insufficient evidence -> abstain rather than fabricate,
- conflicting evidence -> expose the conflict rather than silently invent certainty.

Prompt injection inside retrieved documents is an important security concern, but a full defense is outside this one-day prototype. It may be documented as a limitation/future improvement.

## Model abstraction and testability

Keep the agent runtime separate from the model/provider.

Use a small interface/protocol conceptually similar to:

```python
class Model:
    def decide(self, state: AgentState) -> Action:
        ...
```

The project should support a deterministic scripted/mock model for tests so the agent loop, tool execution, validation, and failure behavior can be tested independently of real-model quality or API availability.

A real-model adapter may be added later, but it should not leak provider-specific logic into the agent runtime.

## State and observability

Keep agent state explicit and small. It should make the run easy to inspect during a live demo.

Useful state concepts may include:

- original question/goal,
- current step number,
- structured actions/tool calls,
- structured tool observations,
- documents actually read,
- constraint-check results,
- final proposal/status.

Do **not** request, expose, persist, or display private model chain-of-thought.

Logs/demo output should show only concise externally meaningful information such as:

- step number,
- chosen action,
- tool name and validated arguments,
- tool result/observation summary,
- validation outcome,
- final answer or abstention reason.

## Evaluation

The finished prototype should include a small transparent evaluation set, approximately 8-10 cases, covering several behaviors rather than only happy paths.

Suggested categories:

- simple factual retrieval,
- multi-source evidence gathering,
- numerical or boolean constraint checking,
- adaptive follow-up/tool choice,
- unanswerable questions requiring abstention,
- at least one failure/safety behavior where practical.

Prefer deterministic, understandable metrics such as:

- task correctness,
- correct abstention,
- evidence/grounding validity,
- valid tool-call rate,
- constraint-check correctness,
- steps/tool calls per run,
- termination within the configured maximum steps.

Do not invent successful evaluation numbers. Report actual results, including failures.

## Scope constraints

Unless the user explicitly changes the scope:

- Use **one agent**, not a multi-agent system.
- Write the main loop in ordinary Python so it is visible and explainable.
- Do **not** use LangChain.
- Do **not** use LangGraph.
- Do **not** use a vector database.
- Do **not** add a web frontend.
- Do **not** add a database unless a concrete need is approved.
- Do **not** add Docker, Kubernetes, cloud infrastructure, authentication, or deployment machinery.
- Do **not** add autonomous web browsing.
- Do **not** add unnecessary design patterns or abstraction layers.
- Keep dependencies minimal.
- Prefer standard-library solutions when they remain clear and safe.

This is intentionally a roughly one-work-day prototype. Unfinished but well-understood work is preferable to unexplained complexity.

## Python and code-quality rules

- Target Python 3.11+.
- Use type hints for public interfaces/functions.
- Prefer dataclasses, typed dictionaries, enums, or small validation models only where they genuinely improve clarity.
- Keep functions/modules focused and short enough to explain live.
- Use clear names rather than clever generic abstractions.
- Use `pytest` for tests.
- Keep tests deterministic where possible.
- Raise/return meaningful errors rather than silently swallowing failures.
- Never commit API keys, tokens, `.env` secrets, or credentials.

## Repository shape

A reasonable target structure is:

```text
research-evidence-agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── data/
│   └── notes/
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── state.py
│       ├── actions.py
│       ├── model.py
│       ├── validation.py
│       └── tools/
│           ├── __init__.py
│           ├── registry.py
│           ├── search.py
│           ├── reader.py
│           ├── calculator.py
│           └── constraints.py
├── tests/
├── eval/
│   ├── cases.json
│   └── evaluate.py
└── demo.py
```

This is a guide, not a requirement to create empty files prematurely. Prefer the smallest structure justified by implemented behavior.

## Codex working rules

Before changing code:

1. Read this file fully.
2. Inspect the existing repository and tests.
3. Restate the requested task briefly and identify the files that need to change.
4. Do not broaden the requested scope without explaining why and obtaining user approval when possible.

After changing code:

1. Run the relevant tests/checks.
2. Report what changed.
3. Explain any dependency added and why it was necessary.
4. Call out unfinished parts or limitations explicitly.
5. Do not perform large unrelated refactors.

When asked to scaffold the project, scaffold only. Do not silently implement later stages of the agent.

## Interview-oriented decision rule

For every nontrivial implementation choice, be prepared to answer:

> Why does the LLM decide this rather than deterministic code?

If deterministic code can perform the operation more reliably and transparently, prefer deterministic code. Use model autonomy where the next action genuinely depends on interpreting the user's goal or observations from previous steps.
