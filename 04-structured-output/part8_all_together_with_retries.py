"""Full structured output pipeline: prompt + JSON mode + Pydantic validation."""

from openai import OpenAI
import json
from pydantic import BaseModel, Field, field_validator

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the library requires something
)


class TaskDescription(BaseModel):
    """Manufacturing task description -- validated schema."""

    task_id: str
    title: str = Field(max_length=100)
    department: str
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(min_length=1)
    estimated_time_minutes: int = Field(ge=1, le=480)
    required_ppe: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v

    @field_validator("steps")
    @classmethod
    def steps_start_with_verb(cls, v):
        for step in v:
            if step and not step[0].isupper():
                raise ValueError(f"Each step should start with a capital letter (action verb): '{step}'")
        return v


def generate_task(task_input: str, max_retries: int = 2) -> TaskDescription | None:
    """Generate a validated task description with retry logic."""
    schema = json.dumps(TaskDescription.model_json_schema(), indent=2)

    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="gemma3:12b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a manufacturing task description generator.\n"
                        f"Return JSON matching this schema:\n{schema}\n\n"
                        "Rules:\n"
                        "- task_id must start with 'TD-' followed by 4 digits\n"
                        "- Each step must start with an action verb (capitalized)\n"
                        "- Include relevant safety requirements and PPE\n"
                        "- Quality checks should be measurable when possible"
                    ),
                },
                {"role": "user", "content": f"Create a task description for: {task_input}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = json.loads(response.choices[0].message.content)

        try:
            task = TaskDescription.model_validate(raw)
            if attempt > 0:
                print(f"  (passed on retry {attempt})")
            return task
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed validation: {e}")
            if attempt == max_retries:
                print("  All retries exhausted.")
                return None

    return None


# --- Try it with several manufacturing tasks ---

tasks = [
    "Inspect hydraulic press cylinder seals for wear and leakage",
    "Calibrate temperature sensors on curing oven",
    "Replace worn conveyor belt rollers",
    "Perform weekly eye wash station inspection",
]

for task_input in tasks:
    print(f"\n{'=' * 60}")
    print(f"Input: {task_input}")
    print(f"{'=' * 60}")

    task = generate_task(task_input)

    if task:
        print(f"  ID:      {task.task_id}")
        print(f"  Title:   {task.title}")
        print(f"  Dept:    {task.department}")
        print(f"  Steps:   {len(task.steps)}")
        for s in task.steps:
            print(f"           - {s}")
        print(f"  Safety:  {task.safety_requirements}")
        print(f"  PPE:     {task.required_ppe}")
        print(f"  Checks:  {task.quality_checks}")
        print(f"  Time:    {task.estimated_time_minutes} min")
    else:
        print("  FAILED -- would need human review")