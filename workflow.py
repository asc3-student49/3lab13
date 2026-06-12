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
from typing import Dict, List, Optional, Any, Callable
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
        agent: Agent,
        prompt_template: str,
        required_inputs: List[str] = None,
        condition: Callable = None,
        retry_count: int = 2,
    ):
        """Add a step to the workflow."""
        step = WorkflowStep(
            name=name,
            agent=agent,
            prompt_template=prompt_template,
            required_inputs=required_inputs or [],
            condition=condition,
            retry_count=retry_count,
        )
        self.steps.append(step)
        return self

    async def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the complete workflow.

        This orchestration loop is provided for you. It handles:
        - Seeding the shared context with initial_data
        - Iterating steps in order and printing progress banners
        - Skipping steps whose condition returns False
        - Validating that each step's required_inputs are present
        - Retrying a step up to retry_count + 1 times on failure
        - Recording final status, errors, and total duration

        It calls self._execute_step(step) to actually run each step — that
        is the method you will implement.
        """
        # Initialize context
        if initial_data:
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
                # Check condition
                if step.condition and not step.condition(self.context):
                    step.status = StepStatus.SKIPPED
                    print(f"⊘ Skipped (condition not met)\n")
                    continue

                # Check required inputs
                missing = [
                    inp for inp in step.required_inputs if not self.context.has(inp)
                ]
                if missing:
                    raise ValueError(f"Missing required inputs: {missing}")

                # Execute step with retries
                step.status = StepStatus.RUNNING
                success = False

                for attempt in range(step.retry_count + 1):
                    step.attempts = attempt + 1

                    try:
                        result = await self._execute_step(step)
                        step.result = result
                        step.status = StepStatus.COMPLETED
                        self.context.steps_completed.append(step.name)
                        success = True
                        print(f"✓ Completed (attempt {attempt + 1})\n")
                        break

                    except Exception as e:
                        if attempt < step.retry_count:
                            print(f"⚠ Attempt {attempt + 1} failed: {e}")
                            print(f"  Retrying...")
                            await asyncio.sleep(1)
                        else:
                            raise

                if not success:
                    raise RuntimeError(
                        f"Step failed after {step.retry_count + 1} attempts"
                    )

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                print(f"✗ Failed: {e}\n")

                # Stop workflow on failure
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
        # YOUR CODE HERE

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


async def main():
    """Main demonstration."""
    print("Workflow Orchestration Examples\n")

    await research_workflow_example()

    print("\n" + "=" * 60 + "\n")

    await conditional_workflow_example()


if __name__ == "__main__":
    asyncio.run(main())
