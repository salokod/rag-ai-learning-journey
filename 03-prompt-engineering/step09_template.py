# 03-prompt-engineering/step09_template.py
import ollama
from string import Template

# --- Your production prompt template ---
# Version control this. Treat it like code, because it IS code.

SYSTEM_PROMPT = """You are a technical writer for a manufacturing facility.
You produce task descriptions following these exact rules:

FORMAT:
- First line: Task title in CAPS
- Second line: blank
- Body: 3-5 numbered steps
- Each step starts with an action verb
- Include tool/equipment references in parentheses where applicable
- Include specification/form references where applicable
- Final step must be a documentation or quality verification step
- Include the department in task title

STYLE:
- Active voice ("Inspect the..." not "The part should be inspected...")
- 8th-grade reading level
- No jargon without definition on first use
- Specific and measurable where possible

SAFETY:
- If the task involves machinery, step 1 must address lockout/tagout or safety
- Always reference required PPE"""

USER_TEMPLATE = Template("""Write a task description for the following:

Task: $task_name
Department: $department
Equipment: $equipment
Relevant specifications: $specifications

Write ONLY the task description, no other commentary.""")


def generate_task_description(task_name, department, equipment, specifications):
    """Generate a standardized manufacturing task description."""
    user_prompt = USER_TEMPLATE.substitute(
        task_name=task_name,
        department=department,
        equipment=equipment,
        specifications=specifications,
    )

    response = ollama.chat(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1, "repeat_penalty": 1.2},
    )
    return response["message"]["content"]


# --- Test it with real manufacturing scenarios ---

test_tasks = [
    {
        "task_name": "Inspect welded joints on Frame Assembly A",
        "department": "Quality Control",
        "equipment": "Visual inspection kit, ultrasonic thickness gauge",
        "specifications": "AWS D1.1, Internal spec WPS-201",
    },
    {
        "task_name": "Set up CNC lathe for shaft machining",
        "department": "Machining",
        "equipment": "Haas ST-20 CNC lathe, tool setter",
        "specifications": "Drawing #SH-4402-Rev.B, tolerance +/-0.005 in",
    },
    {
        "task_name": "Perform daily forklift inspection",
        "department": "Clean Room",
        "equipment": "Toyota 8FGU25 forklift",
        "specifications": "OSHA 1910.178, Company SOP-FL-001",
    },
]

for task in test_tasks:
    print(f"\n{'=' * 60}")
    print(f"Input: {task['task_name']}")
    print(f"{'=' * 60}")
    description = generate_task_description(**task)
    print(description)

print(f"\n{'=' * 60}")
print("WHAT YOU BUILT")
print(f"{'=' * 60}")
print("A reusable, version-controlled prompt template that produces")
print("consistent task descriptions across different departments.")
print("\nNext steps in your journey:")
print("  Module 04 -> Get this output as structured JSON, not just text")
print("  Module 06 -> Feed it real reference documents via RAG")
print("  Module 09 -> Measure quality automatically instead of eyeballing it")