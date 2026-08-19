# Research Evidence Agent

This repository is the initial scaffold for a deliberately small Research
Evidence Agent. The eventual project will demonstrate the core mechanics of an
AI agent in a form that is clear enough to explain during a technical
interview.

The intended architecture is an **observe-decide-act** loop: a model will choose
one next action based on the user's goal and prior observations, application
code will execute that action, and the resulting structured observation will
inform the model's next decision.

Deterministic Python code will be responsible for tool execution, input and
final-answer validation, explicit constraint checks, and safety limits. Model
decisions will be reserved for semantic interpretation and adaptive action
selection.

This task creates the project structure only. The agent loop, tools, model
integration, evidence dataset, and evaluation logic are intentionally not
implemented yet.

## Requirements

- Python 3.11 or newer
- `pytest` for future tests
