from pydantic import BaseModel

class SimpleTask(BaseModel):
    title: str
    steps: list[str]
    time_minutes: int

task = SimpleTask(title="Check seals", steps=["Step 1", "Step 2"], time_minutes=30)
print(task)

task2 = SimpleTask(title="Check seals", steps="not a list", time_minutes=30)
print(task2)
