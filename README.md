# Research Evidence Agent

This repository contains a deliberately small Research Evidence Agent whose
core mechanics are clear enough to explain during a technical interview.

The architecture is an **observe-decide-act** loop: a model chooses
one next action based on the user's goal and prior observations, application
code will execute that action, and the resulting structured observation will
inform the model's next decision.

Deterministic Python code will be responsible for tool execution, input and
final-answer validation, explicit constraint checks, and safety limits. Model
decisions will be reserved for semantic interpretation and adaptive action
selection.

The repository includes a six-note synthetic evidence corpus, four deterministic
tools for lexical note search, note retrieval, safe arithmetic, and
source-backed constraint checking, typed actions and state, a provider-neutral
`DecisionModel`, an explicit single-agent runtime loop, and a deterministic
final-answer provenance gate.

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

No real LLM/API provider, CLI workflow, or evaluation runner is implemented
yet. The provenance gate verifies that cited document IDs were successfully
read during the run; it does not judge whether the answer's claims are true.

## Requirements

- Python 3.11 or newer
- `pytest` for tests
