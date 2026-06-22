# DeepSeek CLI v7.0 — Planner Layer
# Transforms agent from reactive to strategic: User -> Planner -> Task Breakdown -> Execution
# LLM-based task decomposition with step prioritization

import json
import time
from rich.console import Console

console = Console()


class PlanStep:
    """A single step in the execution plan."""

    def __init__(self, index: int, description: str, tool_hint: str = '',
                 priority: str = 'normal', status: str = 'pending'):
        self.index = index
        self.description = description
        self.tool_hint = tool_hint  # Suggested tool category
        self.priority = priority    # high, normal, low
        self.status = status        # pending, in_progress, done, failed, skipped

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'description': self.description,
            'tool_hint': self.tool_hint,
            'priority': self.priority,
            'status': self.status,
        }

    def __repr__(self):
        status_icon = {
            'pending': '[dim]-[/dim]',
            'in_progress': '[bold cyan]>[/bold cyan]',
            'done': '[green]+[/green]',
            'failed': '[red]x[/red]',
            'skipped': '[dim]_[/dim]',
        }.get(self.status, '?')
        return f'  {status_icon} [{self.index}] {self.description}'


class ExecutionPlan:
    """Complete execution plan with steps and metadata."""

    def __init__(self, query: str, steps: list[PlanStep], reasoning: str = ''):
        self.query = query
        self.steps = steps
        self.reasoning = reasoning
        self.created_at = time.time()
        self.completed_steps = 0
        self.failed_steps = 0

    @property
    def is_complete(self) -> bool:
        return all(s.status in ('done', 'skipped') for s in self.steps)

    @property
    def progress_pct(self) -> int:
        if not self.steps:
            return 100
        done = sum(1 for s in self.steps if s.status in ('done', 'skipped'))
        return int(done / len(self.steps) * 100)

    def get_next_pending(self) -> PlanStep | None:
        """Get the next pending step (priority order)."""
        for step in self.steps:
            if step.status == 'pending':
                return step
        return None

    def mark_step(self, index: int, status: str):
        """Mark a step's status."""
        for step in self.steps:
            if step.index == index:
                step.status = status
                if status == 'done':
                    self.completed_steps += 1
                elif status == 'failed':
                    self.failed_steps += 1
                break

    def summarize(self) -> str:
        """Return a summary for display."""
        total = len(self.steps)
        done = self.completed_steps
        failed = self.failed_steps
        pending = total - done - failed
        return (f'Plan: {done}/{total} completed, {pending} pending, {failed} failed '
                f'({self.progress_pct}%)')


class Planner:
    """
    Strategic planner that decomposes user tasks into actionable steps.

    Instead of the old flow: User -> LLM -> Tool -> LLM -> Output
    New flow: User -> Planner -> Task Breakdown -> Smart Execution -> Final Response

    The planner uses a lightweight LLM call to analyze the user's request
    and create a structured plan. This plan guides tool selection and
    execution order, making the agent more strategic and efficient.
    """

    PLANNING_PROMPT = """You are a task planner. Break down the user's request into clear, actionable steps.

Rules:
- Each step should be a SINGLE, specific action
- Order steps logically (dependencies first)
- Use tool hints to suggest which tool category each step needs
- Maximum 6 steps (keep it focused)
- If the task is simple (direct answer, no tools needed), return just 1 step: "Answer directly"
- Be concise — each step description should be 1-2 sentences

Tool categories available:
  file: read/write/list files
  web: web search, fetch URLs
  code: execute code, run shell commands
  system: system info, processes, disk
  math: calculations, unit conversion
  utility: text transform, json, regex, timestamp
  search: live web search, model search
  browser: navigate, click, fill forms
  ocr: read text from images
  mcp: real-time data (weather, stock, news, etc.)

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "reasoning": "Brief reasoning about the approach",
  "steps": [
    {"description": "Step description", "tool_hint": "category", "priority": "high|normal|low"},
    ...
  ]
}"""

    def __init__(self, provider=None):
        self.provider = provider
        self._plan_history: list[ExecutionPlan] = []

    def create_plan(self, user_query: str, message_count: int = 0) -> ExecutionPlan:
        """
        Create an execution plan for the user's query.

        Uses the LLM to decompose the task. Falls back to a simple
        single-step plan if LLM is unavailable.
        """
        # Simple heuristic: skip planning for trivial queries
        if self._is_trivial(user_query):
            return self._simple_plan(user_query)

        # Try LLM-based planning
        if self.provider:
            try:
                plan = self._llm_plan(user_query)
                if plan and plan.steps:
                    self._plan_history.append(plan)
                    return plan
            except Exception:
                pass  # Fall through to simple plan

        return self._simple_plan(user_query)

    def _is_trivial(self, query: str) -> bool:
        """Detect queries that don't need planning."""
        trivial_patterns = [
            '/help', '/exit', '/clear', '/version', '/tools',
            '/info', '/thinking', '/compact', '/export',
        ]
        query_lower = query.strip().lower()
        for p in trivial_patterns:
            if query_lower == p:
                return True
        # Very short queries likely don't need planning
        if len(query.strip()) < 10 and not any(c in query for c in ['?', 'how', 'what', 'why']):
            return True
        return False

    def _simple_plan(self, query: str) -> ExecutionPlan:
        """Create a simple single-step plan (no LLM needed)."""
        step = PlanStep(
            index=0,
            description=f'Process: {query[:100]}',
            tool_hint='',
            priority='normal',
            status='pending',
        )
        return ExecutionPlan(query, [step], reasoning='Direct processing (no decomposition needed)')

    def _llm_plan(self, query: str) -> ExecutionPlan | None:
        """Use LLM to create a structured plan."""
        if not self.provider:
            return None

        # Build a minimal message for planning
        plan_messages = [
            {'role': 'system', 'content': self.PLANNING_PROMPT},
            {'role': 'user', 'content': f'Plan this request: {query}'},
        ]

        # Use a fast, non-streaming call
        response_text = ''
        try:
            for chunk in self.provider.chat_stream(
                messages=plan_messages,
                model=self.provider.default_model,
                temperature=0.3,
                max_tokens=512,
                tools=None,  # No tools for planning
            ):
                if chunk['type'] == 'content':
                    response_text += chunk['data']
                elif chunk['type'] == 'done':
                    break
                elif chunk['type'] == 'error':
                    return None
        except Exception:
            return None

        if not response_text.strip():
            return None

        # Parse the plan JSON
        try:
            # Clean response: remove markdown code blocks if present
            clean = response_text.strip()
            if clean.startswith('```'):
                lines = clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                clean = '\n'.join(lines)
            plan_data = json.loads(clean)
        except json.JSONDecodeError:
            return None

        steps_data = plan_data.get('steps', [])
        if not steps_data:
            return None

        # Limit to 6 steps
        steps_data = steps_data[:6]

        # Build PlanStep objects
        steps = []
        for i, sd in enumerate(steps_data):
            step = PlanStep(
                index=i,
                description=sd.get('description', f'Step {i+1}'),
                tool_hint=sd.get('tool_hint', ''),
                priority=sd.get('priority', 'normal'),
                status='pending',
            )
            steps.append(step)

        reasoning = plan_data.get('reasoning', '')
        return ExecutionPlan(query, steps, reasoning)

    def refine_plan(self, plan: ExecutionPlan, failure_info: str) -> ExecutionPlan:
        """
        Refine a plan after a step failure.
        Tries to adjust remaining steps based on what went wrong.
        """
        # Simple heuristic: mark failed step, skip dependent steps
        for step in plan.steps:
            if step.status == 'failed':
                # Try to adjust the next pending step's priority
                for next_step in plan.steps:
                    if next_step.status == 'pending':
                        next_step.priority = 'high'  # Boost priority
                        break
        return plan

    def get_plan_context(self, plan: ExecutionPlan) -> str:
        """Generate a context string to inject into the agent's system prompt."""
        if not plan or not plan.steps:
            return ''

        lines = ['[Active Plan]']
        for step in plan.steps:
            status_marker = {
                'pending': '[ ]',
                'in_progress': '[>]',
                'done': '[+]',
                'failed': '[x]',
                'skipped': '[_]',
            }.get(step.status, '[?]')

            hint = f' (tool: {step.tool_hint})' if step.tool_hint else ''
            priority = f' [{step.priority}]' if step.priority == 'high' else ''
            lines.append(f'  {status_marker} {step.description}{hint}{priority}')

        lines.append(f'  Progress: {plan.summarize()}')
        return '\n'.join(lines)

    def should_plan(self, query: str, message_count: int = 0) -> bool:
        """Decide whether planning is worthwhile for this query."""
        if self._is_trivial(query):
            return False
        # Don't plan for follow-up messages in an active conversation
        if message_count > 6:
            return False
        return True
