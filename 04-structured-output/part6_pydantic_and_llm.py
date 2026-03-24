from openai import OpenAI
import json
from pydantic import BaseModel, Field, field_validator

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


class TaskDescription(BaseModel):
    task_id: str
    title: str = Field(max_length=100)
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    estimated_time_minutes: int = Field(ge=1, le=480)
    required_ppe: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v


schema = json.dumps(TaskDescription.model_json_schema(), indent=2)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[
        {
            "role": "system",
            "content": f"Generate manufacturing task descriptions as JSON matching this schema:\n{schema}",
        },
        {"role": "user", "content": "Task: Calibrate temperature sensors on curing oven"},
    ],
    response_format={"type": "json_object"},
)

raw = json.loads(response.choices[0].message.content)
print("Raw from LLM:")
print(json.dumps(raw, indent=2))

# try:
#     task = TaskDescription.model_validate(raw)
#     print(f"\nValidation passed!")
#     print(f"  ID:    {task.task_id}")
#     print(f"  Title: {task.title}")
#     print(f"  Steps: {len(task.steps)}")
#     print(f"  PPE:   {task.required_ppe}")
#     print(f"  Time:  {task.estimated_time_minutes} min")
# except Exception as e:
#     print(f"\nValidation FAILED: {e}")
#     print("The LLM returned something that doesn't match our rules.")

bad_outputs = [
    {"task_id": "TASK-99", "title": "Test", "steps": ["one step"], "estimated_time_minutes": 30},
    {"task_id": "TD-0001", "title": "T" * 200, "steps": ["a", "b"], "estimated_time_minutes": 30},
    {"task_id": "TD-0001", "title": "Test", "steps": ["a", "b"], "estimated_time_minutes": 999},
]

for i, bad in enumerate(bad_outputs):
    print(f"\nTest {i + 1}: ", end="")
    try:
        TaskDescription.model_validate(bad)
        print("Passed (unexpected!)")
    except Exception as e:
        print(f"Caught: {e}")