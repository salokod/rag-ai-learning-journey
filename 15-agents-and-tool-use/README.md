# Module 15: Agents & Tool Use

## Goal
Make the LLM stop guessing and start LOOKING THINGS UP. Build tools the model can call, then wire them into a simple agent loop.

---

## The Problem: LLMs Guess When They Should Look It Up

Think about what happens when you ask an LLM for a torque spec. It makes up something plausible. That's terrifying in manufacturing.

What if the model could say: "Hang on, let me look that up for you"?

That's what tool calling is. Let's build it.

---

## Step 1: Build a Tiny "Database"

Before we give the LLM any tools, let's build the data it will look up. Open a Python shell:

```python
python3
```

Now paste this -- it's just a dictionary pretending to be a spec database:

```python
SPECS = {
    "MT-302": "Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH.",
}
```

Try looking something up:

```python
SPECS["MT-302"]
```

You get the exact spec. No hallucination, no guessing. That's what we want the LLM to do.

Now try a spec that doesn't exist:

```python
SPECS.get("FAKE-999", "Not found in database.")
```

Notice how `.get()` returns a safe fallback instead of crashing. We'll need that.

---

## Step 2: Wrap It in a Function

The LLM can't read dictionaries directly. It calls functions. Let's write one:

```python
def lookup_spec(spec_id: str) -> str:
    """Look up a manufacturing specification by ID."""
    return SPECS.get(spec_id.upper(), f"Spec '{spec_id}' not found.")
```

Test it:

```python
lookup_spec("mt-302")
```

Works even with lowercase -- notice the `.upper()` handles that. Now try:

```python
lookup_spec("FAKE-123")
```

Clean error message. Good. The LLM will know the spec doesn't exist instead of making one up.

---

## Step 3: Tell the LLM Your Tool Exists

Here's where it gets interesting. You describe your function to the LLM using a schema. Exit your Python shell and create this file:

```python
# 15-agents-and-tool-use/ex1_first_tool.py
"""Your first tool call -- the LLM decides to look up a spec."""

import ollama

SPECS = {
    "MT-302": "Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH.",
}

def lookup_spec(spec_id: str) -> str:
    return SPECS.get(spec_id.upper(), f"Spec '{spec_id}' not found.")
```

Now add the tool schema -- this tells the LLM what the function does and what arguments it takes:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_spec",
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
]
```

That looks like a lot, but read it carefully. It's just saying: "There's a function called `lookup_spec` that takes one string called `spec_id`." The description is critical -- that's how the model knows WHEN to use it.

---

## Step 4: Ask a Question and Watch the Model Choose

Add this to the same file:

```python
messages = [
    {"role": "system", "content": "You write manufacturing task descriptions. Use tools to look up specs -- never guess."},
    {"role": "user", "content": "What's the torque spec for M10 bolts on Frame #4200?"},
]

response = ollama.chat(
    model="qwen3:32b",
    messages=messages,
    tools=tools,
)

print(response["message"])
```

Run it:

```bash
python3 15-agents-and-tool-use/ex1_first_tool.py
```

Look at the response. The model didn't answer your question. Instead, it returned a `tool_calls` field saying "I want to call `lookup_spec` with `spec_id='MT-302'`."

The model CHOSE to call your function. It decided it needed more information before answering. That's the whole idea.

---

## Step 5: Execute the Tool Call and Feed It Back

The LLM asked to call your function, but it can't actually run code. YOU run it, then give the result back. Add this:

```python
if response["message"].get("tool_calls"):
    tool_call = response["message"]["tool_calls"][0]
    func_name = tool_call["function"]["name"]
    func_args = tool_call["function"]["arguments"]

    print(f"Model wants to call: {func_name}({func_args})")

    # Execute it
    result = lookup_spec(func_args["spec_id"])
    print(f"Tool returned: {result}")

    # Feed the result back to the model
    messages.append(response["message"])
    messages.append({"role": "tool", "content": result})

    # Now the model can answer with real data
    final = ollama.chat(model="qwen3:32b", messages=messages)
    print(f"\nFinal answer:\n{final['message']['content']}")
```

Run it again. This time you should see three things:
1. The model's tool call request
2. The actual spec data from your "database"
3. A final answer that includes the REAL torque values

The answer now includes `M10=45-55Nm` -- the actual spec from your database. No hallucination. The model looked it up.

---

## Step 6: Add More Tools

One tool is useful. Three tools are powerful. Let's give the model a full toolkit. Create a new file:

```python
# 15-agents-and-tool-use/ex2_multi_tool.py
"""Multiple tools -- the model picks which ones it needs."""

import ollama

# Three "databases"
SPECS = {
    "MT-302": "Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Calibrated wrench required.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass max 400F.",
    "AWS-D1.1": "Structural welding code. Visual inspection per Section 6. UT for critical joints.",
}

FORMS = {
    "visual": "Form QC-107: Visual inspection checklist. All items must pass.",
    "dimensional": "Form QC-110: Dimensional inspection. Record actual vs nominal.",
    "weld": "Form QC-115: Weld inspection. References WPS, joint type, NDE results.",
}

PPE = {
    "welding": "Auto-darkening helmet (shade 10-13), leather gloves, FR clothing, steel-toes.",
    "grinding": "Face shield, safety glasses, leather gloves, hearing protection, steel-toes.",
    "machining": "Safety glasses, hearing protection if >85dB, steel-toes. No loose clothing.",
    "general": "Safety glasses and steel-toe boots required in all production areas.",
}

def lookup_spec(spec_id: str) -> str:
    return SPECS.get(spec_id.upper(), f"Spec '{spec_id}' not found.")

def lookup_form(inspection_type: str) -> str:
    return FORMS.get(inspection_type.lower(), f"No form for '{inspection_type}'.")

def get_ppe(task_type: str) -> str:
    return PPE.get(task_type.lower(), PPE["general"])
```

Now define all three tool schemas:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_spec",
            "description": "Look up a manufacturing specification by its ID (e.g., MT-302, WPS-201)",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string", "description": "The specification ID"}
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
                    "inspection_type": {"type": "string", "description": "Type of inspection (visual, dimensional, weld)"}
                },
                "required": ["inspection_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ppe",
            "description": "Get PPE requirements for a specific task type",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "Type of task (welding, grinding, machining, general)"}
                },
                "required": ["task_type"],
            },
        },
    },
]
```

And a helper to execute whichever tool the model calls:

```python
def execute_tool(name: str, args: dict) -> str:
    if name == "lookup_spec":
        return lookup_spec(args["spec_id"])
    elif name == "lookup_form":
        return lookup_form(args["inspection_type"])
    elif name == "get_ppe":
        return get_ppe(args["task_type"])
    return "Unknown tool"
```

---

## Step 7: Ask a Complex Question

Now let's ask something that requires MULTIPLE lookups:

```python
question = "Write a task description for inspecting welds on Frame #4200. Include the torque specs, correct inspection form, and PPE requirements."

messages = [
    {"role": "system", "content": "You are a manufacturing technical writer. Use tools to look up accurate information. Never guess specs, forms, or PPE requirements."},
    {"role": "user", "content": question},
]

print(f"Question: {question}\n")

# First call -- model requests tools
response = ollama.chat(
    model="qwen3:32b",
    messages=messages,
    tools=tools,
    options={"temperature": 0.0},
)

if response["message"].get("tool_calls"):
    messages.append(response["message"])

    print("=== Tool Calls ===")
    for tc in response["message"]["tool_calls"]:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        result = execute_tool(name, args)
        print(f"  {name}({args}) -> {result[:60]}...")
        messages.append({"role": "tool", "content": result})

    # Final answer with all the real data
    final = ollama.chat(
        model="qwen3:32b",
        messages=messages,
        options={"temperature": 0.0},
    )
    print(f"\n=== Final Answer ===\n{final['message']['content']}")
else:
    print("Model answered without tools:")
    print(response["message"]["content"])
```

Run it:

```bash
python3 15-agents-and-tool-use/ex2_multi_tool.py
```

Watch the tool calls. The model should call multiple tools -- `lookup_spec`, `lookup_form`, and `get_ppe` -- all from a single question. It figured out which information it needed.

The final answer has real spec numbers, the correct form, and actual PPE requirements. All from your database, not from the model's imagination.

---

## Step 8: Passing Python Functions Directly (Ollama 0.4+)

If you're running Ollama 0.4 or later, there's a shortcut. Instead of writing tool schemas by hand, you can pass Python functions directly:

```python
# 15-agents-and-tool-use/ex3_direct_functions.py
"""Ollama 0.4+: pass functions directly instead of writing schemas."""

import ollama

SPECS = {
    "MT-302": "Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2.",
}

def lookup_spec(spec_id: str) -> str:
    """Look up a manufacturing specification by its ID (e.g., MT-302, WPS-201)."""
    return SPECS.get(spec_id.upper(), f"Spec '{spec_id}' not found.")

# Pass the function directly -- Ollama builds the schema from your docstring + type hints
response = ollama.chat(
    model="qwen3:32b",
    messages=[
        {"role": "user", "content": "What's the torque spec for Frame #4200?"},
    ],
    tools=[lookup_spec],  # <-- just pass the function!
)

print(response["message"])
```

Notice the difference: `tools=[lookup_spec]` instead of that big JSON schema. Ollama reads your function's name, docstring, and type hints to build the schema automatically.

This is convenient, but the manual schema gives you more control over descriptions. Use whichever you prefer.

---

## Step 9: Build a Simple Agent Loop

So far, you call the LLM once, it requests tools, you execute them, you call it again. That's a single round trip. An agent does this in a LOOP:

```
Think -> Act -> Observe -> Think -> Act -> Observe -> ... -> Done
```

Why? Some questions need multiple rounds of lookup. "Write a complete task description" might require looking up the spec, then the form, then PPE, then realizing it needs a LOTO procedure too.

Let's build it:

```python
# 15-agents-and-tool-use/ex4_agent_loop.py
"""A simple agent loop: think, act, observe, repeat."""

import ollama

# Same databases as before
SPECS = {
    "MT-302": "Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern.",
    "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH.",
    "SOP-SAFE-001": "LOTO: notify, shutdown, isolate, lock/tag, release stored energy, verify zero energy.",
}

FORMS = {
    "visual": "Form QC-107: Visual inspection checklist.",
    "weld": "Form QC-115: Weld inspection per WPS.",
    "torque": "Form QC-110: Dimensional/torque verification.",
}

PPE = {
    "welding": "Auto-darkening helmet, leather gloves, FR clothing, steel-toes.",
    "grinding": "Face shield, safety glasses, hearing protection, steel-toes.",
    "general": "Safety glasses, steel-toe boots.",
}

def execute_tool(name: str, args: dict) -> str:
    if name == "lookup_spec":
        return SPECS.get(args["spec_id"].upper(), f"Spec '{args['spec_id']}' not found.")
    elif name == "lookup_form":
        return FORMS.get(args["inspection_type"].lower(), "Form not found.")
    elif name == "get_ppe":
        return PPE.get(args["task_type"].lower(), PPE["general"])
    return "Unknown tool"
```

Now the tools schema (same as before -- skip if you still have it):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_spec",
            "description": "Look up a manufacturing specification by ID",
            "parameters": {
                "type": "object",
                "properties": {"spec_id": {"type": "string", "description": "The spec ID"}},
                "required": ["spec_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_form",
            "description": "Look up the correct quality form for an inspection type",
            "parameters": {
                "type": "object",
                "properties": {"inspection_type": {"type": "string", "description": "Type: visual, weld, torque"}},
                "required": ["inspection_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ppe",
            "description": "Get PPE requirements for a task type",
            "parameters": {
                "type": "object",
                "properties": {"task_type": {"type": "string", "description": "Type: welding, grinding, general"}},
                "required": ["task_type"],
            },
        },
    },
]
```

And here's the actual agent loop. This is the core idea:

```python
def agent_loop(question: str, max_rounds: int = 5) -> str:
    """Run a think-act-observe loop until the model has enough info to answer."""

    messages = [
        {"role": "system", "content": (
            "You are a manufacturing technical writer. "
            "Use the provided tools to look up every spec, form, and PPE requirement. "
            "NEVER guess -- always look it up. "
            "When you have all the information you need, write the final answer."
        )},
        {"role": "user", "content": question},
    ]

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        response = ollama.chat(
            model="qwen3:32b",
            messages=messages,
            tools=tools,
            options={"temperature": 0.0},
        )

        msg = response["message"]

        # If there are tool calls, execute them and continue the loop
        if msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                result = execute_tool(name, args)
                print(f"  Tool: {name}({args}) -> {result[:70]}")
                messages.append({"role": "tool", "content": result})
        else:
            # No tool calls -- the model is done gathering info and giving its answer
            print(f"  Agent finished after {round_num + 1} round(s).")
            return msg["content"]

    return "Max rounds reached."
```

Try it:

```python
result = agent_loop(
    "Write a complete task description for verifying torque on Frame #4200 per MT-302, "
    "including weld inspection per WPS-201, the correct forms, and all PPE requirements."
)

print(f"\n{'='*60}")
print("FINAL TASK DESCRIPTION:")
print(result)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex4_agent_loop.py
```

Watch the rounds. The model calls tools, gets results, then decides if it needs more information or has enough to write the final answer. It might take 2-3 rounds.

Notice the pattern:
- Round 1: Looks up MT-302 and WPS-201
- Round 2: Looks up the inspection form and PPE
- Round 3: Writes the final task description using ALL the real data

That's the agent loop. Think, act, observe, repeat until done.

---

## Step 10: The Full Manufacturing Agent

Let's put it all together in a clean class. This is the culmination of everything in this module:

```python
# 15-agents-and-tool-use/ex5_manufacturing_agent.py
"""A complete manufacturing agent that gathers info then writes task descriptions."""

import ollama


class ManufacturingAgent:
    """Agent that looks up specs, forms, and PPE before writing task descriptions."""

    def __init__(self, model: str = "qwen3:32b"):
        self.model = model
        self.history = []  # Track what the agent looked up

        # Knowledge bases
        self.specs = {
            "MT-302": "Torque spec Frame #4200: M8=25-30Nm, M10=45-55Nm, M12=80-100Nm. Star pattern. Calibrated wrench ±2%.",
            "WPS-201": "GMAW carbon steel. ER70S-6 wire. 75/25 Ar/CO2 at 25-30 CFH. Interpass max 400F.",
            "AWS-D1.1": "Structural welding code. Visual per Section 6. UT for critical joints. Table 6.1 acceptance.",
            "SOP-SAFE-001": "LOTO: notify operators, normal shutdown, isolate energy, lock/tag, release stored energy, verify zero energy.",
        }
        self.forms = {
            "visual": "Form QC-107: Visual inspection checklist. All items pass or HOLD tag.",
            "dimensional": "Form QC-110: Dimensional inspection. Record actual vs nominal.",
            "weld": "Form QC-115: Weld inspection per WPS. Joint type, position, NDE results.",
            "torque": "Form QC-110: Torque verification. Record spec vs actual values.",
            "calibration": "Form CAL-201: Calibration record. Serial#, standard used, pass/fail.",
        }
        self.ppe = {
            "welding": "Auto-darkening helmet shade 10-13, leather welding gloves, FR clothing, steel-toe boots, safety glasses under helmet.",
            "grinding": "Face shield, safety glasses, leather gloves, hearing protection, steel-toe boots. Grinder guard must be in place.",
            "machining": "Safety glasses, hearing protection if >85dB, steel-toe boots. No loose clothing or jewelry.",
            "press": "Safety glasses, steel-toe boots, hearing protection. Never bypass light curtains or interlocks.",
            "general": "Safety glasses and steel-toe boots required in all production areas.",
        }

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "lookup_spec":
            result = self.specs.get(args["spec_id"].upper(), f"Spec '{args['spec_id']}' not found.")
        elif name == "lookup_form":
            result = self.forms.get(args["inspection_type"].lower(), "Form not found.")
        elif name == "get_ppe":
            result = self.ppe.get(args["task_type"].lower(), self.ppe["general"])
        else:
            result = "Unknown tool"
        self.history.append({"tool": name, "args": args, "result": result})
        return result

    def run(self, task: str, max_rounds: int = 5) -> str:
        """Run the agent loop for a given task."""

        self.history = []

        tools = [
            {"type": "function", "function": {
                "name": "lookup_spec",
                "description": "Look up a manufacturing specification by ID",
                "parameters": {"type": "object", "properties": {"spec_id": {"type": "string", "description": "The spec ID (e.g., MT-302)"}}, "required": ["spec_id"]},
            }},
            {"type": "function", "function": {
                "name": "lookup_form",
                "description": "Look up the correct quality form for an inspection type",
                "parameters": {"type": "object", "properties": {"inspection_type": {"type": "string", "description": "Type: visual, dimensional, weld, torque, calibration"}}, "required": ["inspection_type"]},
            }},
            {"type": "function", "function": {
                "name": "get_ppe",
                "description": "Get PPE requirements for a task type",
                "parameters": {"type": "object", "properties": {"task_type": {"type": "string", "description": "Type: welding, grinding, machining, press, general"}}, "required": ["task_type"]},
            }},
        ]

        messages = [
            {"role": "system", "content": (
                "You are an expert manufacturing technical writer. Your job is to write "
                "accurate, complete task descriptions. ALWAYS use tools to look up specifications, "
                "forms, and PPE requirements. Never guess or make up reference numbers. "
                "Include numbered steps, safety requirements, and all relevant spec references."
            )},
            {"role": "user", "content": task},
        ]

        for round_num in range(max_rounds):
            response = ollama.chat(
                model=self.model, messages=messages, tools=tools,
                options={"temperature": 0.0},
            )
            msg = response["message"]

            if msg.get("tool_calls"):
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    result = self._execute_tool(name, args)
                    messages.append({"role": "tool", "content": result})
            else:
                return msg["content"]

        return "Agent reached max rounds without finishing."


# Let's use it
agent = ManufacturingAgent()

task = (
    "Write a complete task description for verifying torque on Frame Assembly #4200 "
    "per specification MT-302, with weld inspection per AWS-D1.1 and WPS-201. "
    "Include the correct inspection forms and all required PPE."
)

print("=== Manufacturing Agent ===")
print(f"Task: {task}\n")

result = agent.run(task)

print("=== Agent's Tool Usage ===")
for h in agent.history:
    print(f"  {h['tool']}({h['args']}) -> {h['result'][:60]}...")

print(f"\n{'='*60}")
print("FINAL TASK DESCRIPTION:")
print(result)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex5_manufacturing_agent.py
```

Look at the tool usage summary at the end. The agent decided what to look up, looked it up, and then wrote a task description grounded in real data. Every spec number, form reference, and PPE requirement came from your database.

---

## What You Built

Let's recap what just happened:

1. **A tool** -- a Python function the LLM can call to get real data
2. **Tool schemas** -- JSON descriptions so the model knows what tools are available
3. **The round trip** -- model requests a tool call, you execute it, feed the result back
4. **Multiple tools** -- the model picks which ones it needs based on the question
5. **The agent loop** -- think, act, observe, repeat until the model has enough info
6. **A manufacturing agent** -- gathers specs, forms, and PPE, then writes the task description

The key insight: the LLM DECIDES what information it needs. You provide the tools. It calls them. The final output is grounded in real data, not hallucinated.

---

## Takeaways

1. **Tool calling lets LLMs access real data** instead of guessing -- critical for manufacturing accuracy
2. **The model chooses which tools to call** based on the question and tool descriptions
3. **Good tool descriptions matter** -- the model relies on them to decide when to use each tool
4. **Agent loops enable multi-step reasoning** -- look up specs THEN write the description
5. **Your code executes the tools, not the LLM** -- the model just decides what to call

## Next Up

Agents are powerful but need guardrails. What if the model hallucinates a spec number? What if it generates unsafe instructions? Module 16 covers guardrails and safety -- catching bad output before it reaches the shop floor.
