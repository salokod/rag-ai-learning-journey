from pydantic import BaseModel, Field, field_validator

# class TaskDescription(BaseModel):
#     task_id: str
#     title: str
#     steps: list[str]

#     @field_validator("task_id")
#     @classmethod
#     def check_task_id_format(cls, v):
#         if not v.startswith("TD-"):
#             raise ValueError("task_id must start with 'TD-'")
#         return v
    
# TaskDescription(task_id="TD-0001", title="Test", steps=["Step 1"])

# TaskDescription(task_id="TASK-0001", title="Test", steps=["Step 1"])


class TaskDescription(BaseModel):
    task_id: str
    title: str = Field(max_length=100)
    steps: list[str] = Field(min_length=2, max_length=10)
    safety_requirements: list[str] = Field(default_factory=list)
    estimated_time_minutes: int = Field(ge=1, le=480)

    @field_validator("task_id")
    @classmethod
    def check_task_id_format(cls, v):
        if not v.startswith("TD-"):
            raise ValueError("task_id must start with 'TD-'")
        return v

# # Too many steps?
# TaskDescription(
#     task_id="TD-0001",
#     title="Test",
#     steps=["s"] * 15,
#     estimated_time_minutes=30,
# )

# Negative time?
TaskDescription(
    task_id="TD-0001",
    title="Test",
    steps=["Step 1", "Step 2"],
    estimated_time_minutes=-5,
)