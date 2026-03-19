# Module 15: Agents & Tool Use

## Goal
Build LLM agents that can use tools — look up information, query databases, run calculations, and chain multiple reasoning steps. Move from "LLM generates text" to "LLM takes actions."

---

## Concepts

### What Is an Agent?

A basic LLM takes input → generates output. An **agent** has a loop:

```
┌──────────────────────────────────────────┐
│                AGENT LOOP                 │
│                                           │
│  1. Observe (read input/tool results)     │
│  2. Think (decide what to do next)        │
│  3. Act (call a tool OR give final answer)│
│  4. → Back to 1 if not done              │
│                                           │
└──────────────────────────────────────────┘
```

### Why Agents for Manufacturing?

Imagine an agent that can:
- **Look up** the correct specification for a part number
- **Query** the calibration database to check if a tool is current
- **Calculate** torque values based on bolt size and material
- **Generate** the task description with all correct references
- **Validate** the output against your quality rubric

That's more useful than a static prompt.

### Tool Calling (Function Calling)

Modern LLMs can decide to "call" predefined functions. You define tools with descriptions, and the model decides when to use them.

---

## Exercise 1: Basic Tool Calling with Ollama

```python
# 15-agents-and-tool-use/ex1_tool_calling.py
"""Give an LLM tools to look up manufacturing information."""

import ollama
import json

# Define tools (functions the LLM can call)
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_specification",
            "description": "Look up a manufacturing specification by its ID (e.g., MT-302, WPS-201)",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "The specification ID to look up",
                    }
                },
                "required": ["spec_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_form",
            "description": "Look up the correct quality form for a given inspection type",
            "parameters": {
                "type": "object",
                "properties": {
                    "inspection_type": {
                        "type": "string",
                        "description": "Type of inspection (visual, dimensional, weld, calibration)",
                    }
                },
                "required": ["inspection_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ppe_requirements",
            "description": "Get PPE requirements for a specific task type",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of task (welding, grinding, machining, press, general)",
                    }
                },
                "required": ["task_type"],
            },
        },
    },
]

# Tool implementations (your "database")
SPECS_DB = {
    "MT-302": "Torque spec for Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Calibrated wrench ±2%.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. No preheat <1\". Interpass max 400°F.",
    "AWS-D1.1": "Structural welding code. Visual inspection Section 6. UT for critical joints. Acceptance criteria per Table 6.1.",
}

FORMS_DB = {
    "visual": "Form QC-107: Visual inspection checklist. Fields: part#, lot#, inspector ID, date. All items must pass.",
    "dimensional": "Form QC-110: Dimensional inspection report. Record actual vs. nominal for each dimension.",
    "weld": "Form QC-115: Weld-specific inspection. References WPS, joint type, position, NDE results.",
    "calibration": "Form CAL-201: Calibration record. Serial#, standard used, readings, pass/fail, next due date.",
}

PPE_DB = {
    "welding": "Auto-darkening helmet (shade 10-13), leather welding gloves, FR clothing, steel-toe boots, safety glasses under helmet.",
    "grinding": "Face shield, safety glasses, leather gloves, hearing protection, steel-toe boots. Ensure grinder guard is in place.",
    "machining": "Safety glasses, hearing protection if >85dB, steel-toe boots. No loose clothing, tie back long hair.",
    "press": "Safety glasses, steel-toe boots, hearing protection. Never bypass light curtains or interlocks.",
    "general": "Safety glasses, steel-toe boots required in all production areas.",
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result."""
    if name == "lookup_specification":
        spec_id = arguments["spec_id"].upper()
        return SPECS_DB.get(spec_id, f"Specification '{spec_id}' not found in database.")
    elif name == "lookup_form":
        insp_type = arguments["inspection_type"].lower()
        return FORMS_DB.get(insp_type, f"No form found for '{insp_type}' inspection.")
    elif name == "get_ppe_requirements":
        task_type = arguments["task_type"].lower()
        return PPE_DB.get(task_type, PPE_DB["general"])
    return "Unknown tool"


# Use tools in a conversation
question = "Write a task description for inspecting welds on Frame #4200. Include the correct torque specs, inspection form, and PPE."

print(f"User: {question}\n")

messages = [
    {"role": "system", "content": "You are a manufacturing technical writer. Use the provided tools to look up accurate information before writing task descriptions."},
    {"role": "user", "content": question},
]

# Step 1: LLM decides which tools to call
response = ollama.chat(
    model="llama3.1:8b",
    messages=messages,
    tools=tools,
    options={"temperature": 0.0},
)

# Step 2: Execute tool calls
if response["message"].get("tool_calls"):
    print("=== Tool Calls ===")
    messages.append(response["message"])

    for tool_call in response["message"]["tool_calls"]:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        print(f"  Calling: {func_name}({func_args})")

        result = execute_tool(func_name, func_args)
        print(f"  Result: {result[:80]}...")

        messages.append({
            "role": "tool",
            "content": result,
        })

    # Step 3: LLM generates final answer with tool results
    print("\n=== Final Response ===")
    final_response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        options={"temperature": 0.0},
    )
    print(final_response["message"]["content"])
else:
    print("Model didn't use tools:")
    print(response["message"]["content"])
```

---

## Exercise 2: Building a Simple Agent Loop

```python
# 15-agents-and-tool-use/ex2_agent_loop.py
"""Build a simple agent that reasons and acts in a loop."""

import ollama
import json
import re


class ManufacturingAgent:
    """An agent that generates task descriptions by looking up references."""

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.max_iterations = 5
        self.tools_used = []

        # Knowledge base
        self.specs = {
            "MT-302": "Torque spec Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm",
            "WPS-201": "GMAW welding: ER70S-6, 75/25 Ar/CO2, interpass 400°F max",
            "SOP-SAFE-001": "LOTO: notify, shutdown, isolate, lock/tag, release energy, verify",
        }
        self.forms = {
            "visual": "QC-107",
            "dimensional": "QC-110",
            "weld": "QC-115",
            "torque": "QC-110",
        }

    def think_and_act(self, task: str) -> str:
        """Agent loop: think → act → observe → repeat."""

        system = """You are a manufacturing agent. Given a task, decide what information
you need, then generate the task description.

Available actions:
- LOOKUP_SPEC: <spec_id> — Look up a specification
- LOOKUP_FORM: <inspection_type> — Find the correct form
- GENERATE: Generate the final task description using gathered info
- DONE: You've completed the task

Always LOOKUP relevant specs and forms before GENERATE.
Format each response as:
THOUGHT: <your reasoning>
ACTION: <action name>: <parameters>"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Create a task description for: {task}"},
        ]

        gathered_info = []

        for iteration in range(self.max_iterations):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.0},
            )

            reply = response["message"]["content"]
            messages.append({"role": "assistant", "content": reply})

            print(f"\n--- Iteration {iteration + 1} ---")
            print(reply[:300])

            # Parse action
            if "DONE" in reply or "GENERATE" in reply:
                # Extract the generated content
                if "GENERATE" in reply:
                    # Ask for final output
                    messages.append({
                        "role": "user",
                        "content": f"Now generate the final task description using this information:\n{chr(10).join(gathered_info)}",
                    })
                    final = ollama.chat(
                        model=self.model,
                        messages=messages,
                        options={"temperature": 0.0},
                    )
                    return final["message"]["content"]
                return reply

            elif "LOOKUP_SPEC" in reply:
                # Extract spec ID
                match = re.search(r'LOOKUP_SPEC[:\s]+([A-Za-z0-9-]+)', reply)
                if match:
                    spec_id = match.group(1).upper()
                    result = self.specs.get(spec_id, f"Spec {spec_id} not found")
                    gathered_info.append(f"Spec {spec_id}: {result}")
                    messages.append({"role": "user", "content": f"OBSERVATION: {result}"})
                    self.tools_used.append(f"LOOKUP_SPEC:{spec_id}")

            elif "LOOKUP_FORM" in reply:
                match = re.search(r'LOOKUP_FORM[:\s]+(\w+)', reply)
                if match:
                    form_type = match.group(1).lower()
                    result = self.forms.get(form_type, "Form not found")
                    gathered_info.append(f"Form for {form_type}: {result}")
                    messages.append({"role": "user", "content": f"OBSERVATION: Form is {result}"})
                    self.tools_used.append(f"LOOKUP_FORM:{form_type}")

            else:
                messages.append({"role": "user", "content": "Please specify an ACTION."})

        return "Max iterations reached"


# Run the agent
agent = ManufacturingAgent()
print("=== Manufacturing Agent ===")
result = agent.think_and_act(
    "Verify torque on Frame Assembly #4200 per specification MT-302, with weld inspection"
)
print(f"\n{'='*60}")
print("FINAL OUTPUT:")
print(result)
print(f"\nTools used: {agent.tools_used}")
```

---

## Takeaways

1. **Tool calling** lets LLMs access real data instead of guessing — critical for manufacturing accuracy
2. **Agent loops** enable multi-step reasoning — look up specs, THEN write the description
3. **Define tools clearly** — good tool descriptions help the model use them correctly
4. **Validate tool results** — the LLM decides WHAT to look up, your code returns CORRECT data
5. **Agents are powerful but complex** — more steps = more chances for errors. Test thoroughly!

## Setting the Stage for Module 16

Agents are powerful but need guardrails. What if the LLM generates unsafe instructions? What if it hallucinates a spec number? Module 16 covers **guardrails and safety** — input validation, output filtering, and hallucination detection to keep your system safe.
