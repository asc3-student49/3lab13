"""
Multi-Step Agent Workflow Orchestration

Coordinates complex workflows with multiple sequential steps:
- Step-by-step execution with state management
- Conditional branching based on step results
- Data passing between steps
- Error recovery and retry logic
- Workflow visualization and logging

Run: python workflow.py
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Literal, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pydantic_ai import Agent
import os
from dotenv import load_dotenv

load_dotenv()


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in the workflow."""

    name: str
    agent: Agent
    prompt_template: str
    required_inputs: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None  # Skip if condition returns False
    retry_count: int = 2
    handler: Optional[Callable[["WorkflowContext"], Awaitable[Any]]] = None

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class WorkflowContext:
    """Shared state across workflow steps."""

    data: Dict[str, Any] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def set(self, key: str, value: Any):
        """Store data."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data


class WorkflowOrchestrator:
    """
    Orchestrates multi-step agent workflows.

    Features:
    - Sequential step execution
    - State management across steps
    - Conditional branching
    - Error handling and retries
    - Execution logging
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.context = WorkflowContext()

    def add_step(
        self,
        name: str,
        agent: Optional[Agent] = None,
        prompt_template: str = "",
        required_inputs: List[str] = None,
        condition: Callable = None,
        retry_count: int = 2,
        handler: Optional[Callable[[WorkflowContext], Awaitable[Any]]] = None,
    ):
        """Add a step to the workflow."""
        if handler is None and (agent is None or not prompt_template):
            raise ValueError(
                "Step requires either a custom handler or both agent and prompt_template"
            )

        step = WorkflowStep(
            name=name,
            agent=agent,
            prompt_template=prompt_template,
            required_inputs=required_inputs or [],
            condition=condition,
            retry_count=retry_count,
            handler=handler,
        )
        self.steps.append(step)
        return self

    async def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the complete workflow.
        """
        if initial_data is not None:
            self.context.data.update(initial_data)

        self.context.start_time = datetime.now()

        print(f"\n{'='*60}")
        print(f"  Workflow: {self.name}")
        print(f"{'='*60}\n")

        # Execute each step
        for i, step in enumerate(self.steps, 1):
            print(f"Step {i}/{len(self.steps)}: {step.name}")
            print(f"{'─'*60}")

            try:
                # Skip when condition exists and returns False.
                if step.condition and not step.condition(self.context):
                    step.status = StepStatus.SKIPPED
                    print(f"⊘ Skipped (condition not met)\n")
                    continue

                # Validate all required inputs exist in shared context.
                missing = [
                    inp for inp in step.required_inputs if not self.context.has(inp)
                ]
                if missing:
                    raise ValueError(f"Missing required inputs: {missing}")

                step.status = StepStatus.RUNNING

                for attempt in range(step.retry_count + 1):
                    step.attempts += 1

                    try:
                        result = await self._execute_step(step)
                        step.result = result
                        step.status = StepStatus.COMPLETED
                        self.context.steps_completed.append(step.name)
                        print(f"✓ Completed (attempt {attempt + 1})\n")
                        break

                    except Exception as e:
                        if attempt < step.retry_count:
                            print(f"⚠ Attempt {attempt + 1} failed: {e}")
                            print("  Retrying...")
                            await asyncio.sleep(1)
                        else:
                            raise

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                print(f"✗ Failed: {e}\n")

                self.context.end_time = datetime.now()
                return self._get_results()

        self.context.end_time = datetime.now()

        duration = (self.context.end_time - self.context.start_time).total_seconds()
        print(f"{'='*60}")
        print(f"  Workflow Completed in {duration:.1f}s")
        print(f"{'='*60}\n")

        return self._get_results()

    async def _execute_step(self, step: WorkflowStep) -> str:
        """Execute a single step.

        ``str.format`` raises ``KeyError`` if ``step.prompt_template``
        references a placeholder that isn't in ``self.context.data``.
        ``execute()`` validates ``step.required_inputs`` before reaching
        here, but does *not* parse the template — so a placeholder
        added in a Challenge extension without a matching
        ``required_inputs`` entry would surface as an unhelpful
        bare ``KeyError``. The except branch below converts that into
        a diagnostic naming both the step and the missing key.
        """
        if step.handler is not None:
            result = await step.handler(self.context)
            normalized = self._normalize_step_result(result)
            self.context.set(f"step_{step.name}_result", normalized)
            return normalized

        if step.agent is None:
            raise ValueError(f"Step '{step.name}' has no agent configured")

        try:
            prompt = step.prompt_template.format(**self.context.data)
        except KeyError as e:
            missing_key = e.args[0]
            raise KeyError(
                f"Step '{step.name}' prompt is missing context key '{missing_key}'. "
                "Add it to required_inputs or ensure a prior step sets it."
            ) from e

        result = await step.agent.run(prompt)
        self.context.set(f"step_{step.name}_result", result.output)
        return result.output

    def _normalize_step_result(self, result: Any) -> str:
        """Normalize arbitrary step output into a string for consistent storage."""
        if isinstance(result, str):
            return result
        if result is None:
            return ""
        return json.dumps(result, ensure_ascii=True)

    def _get_results(self) -> Dict[str, Any]:
        """Get workflow results."""
        return {
            "status": (
                "completed"
                if all(
                    s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                    for s in self.steps
                )
                else "failed"
            ),
            "context": self.context.data,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                    "attempts": s.attempts,
                }
                for s in self.steps
            ],
            "duration_seconds": (
                (self.context.end_time - self.context.start_time).total_seconds()
                if self.context.end_time
                else None
            ),
        }


TaskCategory = Literal["research", "execution"]


def classify_task(task_input: str) -> TaskCategory:
    """Classify a task into a workflow category.

    This is intentionally lightweight and rule-based so the factory can
    select a step graph before any expensive agent calls run.
    """
    text = task_input.lower()

    research_keywords = {
        "research",
        "analyze",
        "analysis",
        "compare",
        "trend",
        "report",
        "study",
        "market",
        "findings",
    }
    execution_keywords = {
        "build",
        "implement",
        "create",
        "deploy",
        "fix",
        "plan",
        "launch",
        "workflow",
        "checklist",
    }

    research_score = sum(1 for keyword in research_keywords if keyword in text)
    execution_score = sum(1 for keyword in execution_keywords if keyword in text)

    # Longer inputs are often exploratory and benefit from research synthesis.
    if len(task_input.split()) >= 20:
        research_score += 1

    return "research" if research_score >= execution_score else "execution"


def build_workflow_factory(category: TaskCategory) -> WorkflowOrchestrator:
    """Create a WorkflowOrchestrator with category-specific steps."""
    model_name = os.getenv("AI_MODEL", "openai:gpt-5.4-mini")

    analyst_agent = Agent(
        model_name,
        system_prompt="You analyze requests and identify key requirements.",
    )
    researcher_agent = Agent(
        model_name,
        system_prompt="You gather and structure high-value research insights.",
    )
    planner_agent = Agent(
        model_name,
        system_prompt="You create actionable plans and clear deliverables.",
    )

    workflow = WorkflowOrchestrator(f"Dynamic Workflow ({category})")

    if category == "research":
        workflow.add_step(
            name="scope_question",
            agent=analyst_agent,
            prompt_template=(
                "Scope this research request and list the main questions: {task_input}"
            ),
            required_inputs=["task_input"],
        )

        workflow.add_step(
            name="collect_findings",
            agent=researcher_agent,
            prompt_template=(
                "Generate key factual findings for this scope:\n\n{step_scope_question_result}"
            ),
            required_inputs=["step_scope_question_result"],
        )

        workflow.add_step(
            name="synthesize_brief",
            agent=planner_agent,
            prompt_template=(
                "Create a concise research brief with recommendations:\n\n"
                "{step_collect_findings_result}"
            ),
            required_inputs=["step_collect_findings_result"],
        )
    else:
        workflow.add_step(
            name="extract_requirements",
            agent=analyst_agent,
            prompt_template="Extract concrete requirements from: {task_input}",
            required_inputs=["task_input"],
        )

        workflow.add_step(
            name="build_action_plan",
            agent=planner_agent,
            prompt_template=(
                "Create a step-by-step execution plan from these requirements:\n\n"
                "{step_extract_requirements_result}"
            ),
            required_inputs=["step_extract_requirements_result"],
        )

        workflow.add_step(
            name="draft_deliverable",
            agent=planner_agent,
            prompt_template=(
                "Draft a delivery-ready response for this request:\n\n"
                "Original task: {task_input}\n\n"
                "Plan:\n{step_build_action_plan_result}"
            ),
            required_inputs=["task_input", "step_build_action_plan_result"],
        )

    return workflow


# Example: Research workflow
async def research_workflow_example():
    """Demonstrate multi-step research workflow."""

    # Create specialized agents
    research_agent = Agent(
        os.getenv("AI_MODEL", "openai:gpt-5.4-mini"),
        system_prompt="You are a research assistant. Provide detailed, factual information.",
    )

    summarizer_agent = Agent(
        os.getenv("AI_MODEL", "openai:gpt-5.4-mini"),
        system_prompt="You are a summarizer. Create concise, clear summaries.",
    )

    outline_agent = Agent(
        os.getenv("AI_MODEL", "openai:gpt-5.4-mini"),
        system_prompt="You are an outline creator. Structure information clearly.",
    )

    # Build workflow
    workflow = WorkflowOrchestrator("Research Report Generation")

    workflow.add_step(
        name="research",
        agent=research_agent,
        prompt_template="Research the topic: {topic}. Provide detailed information.",
        required_inputs=["topic"],
    )

    workflow.add_step(
        name="summarize",
        agent=summarizer_agent,
        prompt_template="Summarize this research:\n\n{step_research_result}",
        required_inputs=["step_research_result"],
    )

    workflow.add_step(
        name="create_outline",
        agent=outline_agent,
        prompt_template="Create an outline for a report based on:\n\n{step_summarize_result}",
        required_inputs=["step_summarize_result"],
    )

    # Execute
    results = await workflow.execute(initial_data={"topic": "artificial intelligence"})

    # Display results
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))


# Example: Conditional workflow
async def conditional_workflow_example():
    """Demonstrate workflow with conditional steps."""

    agent = Agent(os.getenv("AI_MODEL", "openai:gpt-5.4-mini"))

    workflow = WorkflowOrchestrator("Order Processing")

    workflow.add_step(
        name="validate_order",
        agent=agent,
        prompt_template="Validate this order: {order}. Respond with VALID or INVALID.",
        required_inputs=["order"],
    )

    # Conditional step - only run if order is valid
    def order_is_valid(ctx):
        result = ctx.get("step_validate_order_result", "")
        return "VALID" in result.upper()

    workflow.add_step(
        name="process_payment",
        agent=agent,
        prompt_template="Process payment for order {order}",
        required_inputs=["order"],
        condition=order_is_valid,
    )

    workflow.add_step(
        name="send_confirmation",
        agent=agent,
        prompt_template="Generate confirmation email for processed order",
        condition=order_is_valid,
    )

    # Execute
    results = await workflow.execute(initial_data={"order": "Book x2, $29.99"})


async def dynamic_factory_workflow_example():
    """Demonstrate dynamic workflow construction from task classification."""
    demo_inputs = [
        "Research emerging trends in edge AI for healthcare diagnostics and summarize findings.",
        "Create a launch checklist and implementation plan for deploying a customer support chatbot.",
    ]

    print("Dynamic Workflow Factory Demo\n")

    for idx, task_input in enumerate(demo_inputs, start=1):
        category = classify_task(task_input)
        workflow = build_workflow_factory(category)

        print(f"Scenario {idx}")
        print(f"Input: {task_input}")
        print(f"Classified category: {category}")
        print(f"Step sequence: {[step.name for step in workflow.steps]}\n")

        results = await workflow.execute(
            initial_data={
                "task_input": task_input,
                "task_category": category,
            }
        )

        executed_path = [
            step["name"]
            for step in results["steps"]
            if step["status"] in ("completed", "running")
        ]
        print(f"Executed path: {executed_path}")
        print(f"Workflow status: {results['status']}")
        print("\n" + "-" * 60 + "\n")


async def nested_subworkflow_example():
    """Demonstrate a parent step invoking a child workflow as a single step."""
    model_name = os.getenv("AI_MODEL", "openai:gpt-5.4-mini")
    planner_agent = Agent(
        model_name,
        system_prompt="You create concise execution artifacts.",
    )
    reviewer_agent = Agent(
        model_name,
        system_prompt="You identify key risks and mitigations.",
    )

    parent_workflow = WorkflowOrchestrator("Parent Workflow With Sub-Workflow Step")

    parent_workflow.add_step(
        name="intake_request",
        agent=planner_agent,
        prompt_template="Extract delivery objective and constraints from: {task_input}",
        required_inputs=["task_input"],
    )

    async def run_child_workflow(ctx: WorkflowContext) -> Dict[str, Any]:
        child = WorkflowOrchestrator("Child Workflow: Plan + Risk")
        child.add_step(
            name="build_plan",
            agent=planner_agent,
            prompt_template=(
                "Create an actionable plan from this request context:\n\n"
                "{step_intake_request_result}"
            ),
            required_inputs=["step_intake_request_result"],
        )
        child.add_step(
            name="assess_risks",
            agent=reviewer_agent,
            prompt_template=(
                "Assess top risks and mitigations for this plan:\n\n"
                "{step_build_plan_result}"
            ),
            required_inputs=["step_build_plan_result"],
        )

        child_results = await child.execute(
            initial_data={"step_intake_request_result": ctx.get("step_intake_request_result")}
        )

        # Merge selected child outputs under one parent key.
        merged_child_data = {
            "status": child_results["status"],
            "plan": child_results["context"].get("step_build_plan_result"),
            "risks": child_results["context"].get("step_assess_risks_result"),
            "duration_seconds": child_results.get("duration_seconds"),
        }
        ctx.set("child_workflow", merged_child_data)

        if child_results["status"] != "completed":
            failed_steps = [
                step["name"]
                for step in child_results["steps"]
                if step["status"] == "failed"
            ]
            partial = {
                "status": "partial",
                "failed_steps": failed_steps,
                "child_workflow": merged_child_data,
            }

            # Failure policy:
            # - default: raise to fail parent step (and allow parent retries)
            # - allow_child_partial=True: record partial output and continue
            if ctx.get("allow_child_partial", False):
                return partial

            raise RuntimeError(
                f"Child workflow failed in steps: {failed_steps or ['unknown']}"
            )

        return {
            "status": "completed",
            "child_workflow": merged_child_data,
        }

    parent_workflow.add_step(
        name="run_child_workflow",
        required_inputs=["step_intake_request_result"],
        retry_count=1,
        handler=run_child_workflow,
    )

    parent_workflow.add_step(
        name="finalize_response",
        agent=planner_agent,
        prompt_template=(
            "Create final delivery using this parent context data:\n\n"
            "Child summary: {child_workflow}\n\n"
            "Child execution result: {step_run_child_workflow_result}"
        ),
        required_inputs=["child_workflow", "step_run_child_workflow_result"],
    )

    results = await parent_workflow.execute(
        initial_data={
            "task_input": "Build a rollout plan for a support chatbot with key risks.",
            "allow_child_partial": False,
        }
    )

    print("Nested Workflow Results:")
    print(json.dumps(results, indent=2))


async def main():
    """Main demonstration."""
    print("Workflow Orchestration Examples\n")

    await research_workflow_example()

    print("\n" + "=" * 60 + "\n")

    await conditional_workflow_example()

    print("\n" + "=" * 60 + "\n")

    await dynamic_factory_workflow_example()

    print("\n" + "=" * 60 + "\n")

    await nested_subworkflow_example()


if __name__ == "__main__":
    asyncio.run(main())
