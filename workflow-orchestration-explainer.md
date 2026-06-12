# Workflow Orchestration Explained

## 1) How orchestration differs from single-agent prompting

Single-agent prompting asks one model to handle everything in one pass. That can work for simple tasks, but it often mixes planning, execution, validation, and formatting in the same response.

Workflow orchestration splits work into explicit steps, each with a clear responsibility.

- Single-agent prompting:
  - One prompt, one model run, one output
  - Limited control over intermediate logic
  - Harder to retry only the failed part
- Orchestration:
  - Multiple named steps with clear boundaries
  - Step-level retries, conditions, and required inputs
  - Better observability because each step has status, attempts, result, and error

In this repo, orchestration is represented by WorkflowOrchestrator, WorkflowStep, and WorkflowContext. This gives you a state machine instead of a single model call.

## 2) How shared context enables data flow between specialized agents

Shared context acts like a pipeline memory that all steps can read from and write to.

- A step reads required keys from context.data
- It runs specialized logic (agent prompt or custom handler)
- It writes its output back under deterministic keys
- Downstream steps consume those keys

This allows specialization without losing continuity. Example pattern:

- Step A (analyst): extract requirements
- Step B (planner): build plan from Step A result
- Step C (reviewer): assess risks from Step B result

The parent-child example also shows context merging:

- A child workflow executes independently
- Selected child outputs are merged into the parent under one logical key: child_workflow
- Parent steps then use child_workflow as a normalized input

## 3) How state and dependency validation improve reliability

Reliability improves when the orchestrator enforces correctness before work runs.

Key reliability controls in this project:

- Required input validation:
  - Each step declares required_inputs
  - Execution fails early if required keys are missing
- Prompt variable safety:
  - Missing template keys are converted into explicit diagnostic errors
- State tracking:
  - Step statuses (pending, running, completed, failed, skipped) are explicit
  - Attempts and errors are recorded
- Controlled retries:
  - Only failed steps are retried up to retry_count
  - Retries avoid rerunning the entire pipeline when unnecessary
- Deterministic result collection:
  - Final results include per-step metadata and total workflow duration

Together, these controls reduce silent failures, improve debuggability, and make behavior predictable.

## 4) How parallel or nested workflows help scale complex pipelines

As pipelines grow, a flat sequence becomes hard to maintain. Parallel and nested designs improve scale.

### Parallel workflows

Use parallelism when steps are independent.

- Run independent branches at the same time
- Reduce end-to-end latency
- Merge branch outputs into shared context when all complete

Typical use cases:

- Multiple research fetches in parallel
- Independent evaluations (quality, safety, style)
- Fan-out processing of chunks followed by aggregation

### Nested workflows

Use nesting when a sub-process is logically one parent step but internally complex.

- Parent step calls a child orchestrator execute()
- Child manages its own internal steps, retries, and failures
- Parent receives a normalized child result and stores it under one key

Benefits:

- Encapsulation: child workflow logic stays modular
- Reuse: same child workflow can be called from multiple parents
- Fault policy control: parent can fail, retry, or continue with partial child output

In this repo, the nested example demonstrates exactly this model with a parent step that invokes a child workflow and merges selected child fields into parent context.

## Practical takeaway

Orchestration turns agent work into a robust pipeline:

- Specialized steps instead of one monolithic prompt
- Shared context for explicit data flow
- Validation and state controls for reliability
- Parallel and nested composition for scale and maintainability
