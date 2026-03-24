# Module 15: Agents & Tool Use

## Goal
Make the LLM stop guessing and start LOOKING THINGS UP. Build tools the model can call, then wire them into a simple agent loop.

---

## The Problem: LLMs Guess When They Should Look It Up

Think about what happens when you ask an LLM for a player's combine numbers. It makes up something plausible. That's terrifying in football scouting.

What if the model could say: "Hang on, let me look that up for you"?

That's what tool calling is. Let's build it.

---

## Step 1: Build a Tiny "Database"

Before we give the LLM any tools, let's build the data it will look up. Open a Python shell:

```python
python3
```

Now paste this -- it's just a dictionary pretending to be a scouting database:

```python
PLAYERS = {
    "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Excels on intermediate routes (15-25 yards). Reads defenses pre-snap. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
    "RB-201": "Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup.",
}
```

Try looking something up:

```python
PLAYERS["QB-101"]
```

You get the exact scouting report. No hallucination, no guessing. That's what we want the LLM to do.

Now try a player that doesn't exist:

```python
PLAYERS.get("FAKE-999", "Not found in database.")
```

Notice how `.get()` returns a safe fallback instead of crashing. We'll need that.

---

## Step 2: Wrap It in a Function

The LLM can't read dictionaries directly. It calls functions. Let's write one:

```python
def lookup_player_stats(player_id: str) -> str:
    """Look up a football player's scouting report by ID."""
    return PLAYERS.get(player_id.upper(), f"Player '{player_id}' not found.")
```

Test it:

```python
lookup_player_stats("qb-101")
```

Works even with lowercase -- notice the `.upper()` handles that. Now try:

```python
lookup_player_stats("FAKE-123")
```

Clean error message. Good. The LLM will know the player doesn't exist instead of making one up.

---

## Step 3: Tell the LLM Your Tool Exists

Here's where it gets interesting. You describe your function to the LLM using a schema. Exit your Python shell and create this file:

```python
# 15-agents-and-tool-use/ex1_first_tool.py
"""Your first tool call -- the LLM decides to look up a player."""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

PLAYERS = {
    "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Excels on intermediate routes (15-25 yards). Reads defenses pre-snap. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
    "RB-201": "Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup.",
}

def lookup_player_stats(player_id: str) -> str:
    return PLAYERS.get(player_id.upper(), f"Player '{player_id}' not found.")
```

Now add the tool schema -- this tells the LLM what the function does and what arguments it takes:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_player_stats",
            "description": "Look up a football player's scouting report by their ID (e.g., QB-101, RB-201)",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {
                        "type": "string",
                        "description": "The player ID to look up",
                    }
                },
                "required": ["player_id"],
            },
        },
    },
]
```

That looks like a lot, but read it carefully. It's just saying: "There's a function called `lookup_player_stats` that takes one string called `player_id`." The description is critical -- that's how the model knows WHEN to use it.

---

## Step 4: Ask a Question and Watch the Model Choose

Add this to the same file:

```python
messages = [
    {"role": "system", "content": "You write football scouting reports. Use tools to look up player data -- never guess."},
    {"role": "user", "content": "What's the arm strength for the QB prospect QB-101?"},
]

response = client.chat.completions.create(
    model="llama3.3:70b",
    messages=messages,
    tools=tools,
)

print(response.choices[0].message)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex1_first_tool.py
```

Look at the response. The model didn't answer your question. Instead, it returned a `tool_calls` field saying "I want to call `lookup_player_stats` with `player_id='QB-101'`."

The model CHOSE to call your function. It decided it needed more information before answering. That's the whole idea.

---

## Step 5: Execute the Tool Call and Feed It Back

The LLM asked to call your function, but it can't actually run code. YOU run it, then give the result back. Add this:

```python
msg = response.choices[0].message
if msg.tool_calls:
    tool_call = msg.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"Model wants to call: {func_name}({func_args})")

    # Execute it
    result = lookup_player_stats(func_args["player_id"])
    print(f"Tool returned: {result}")

    # Feed the result back to the model
    messages.append(msg)
    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})

    # Now the model can answer with real data
    final = client.chat.completions.create(model="llama3.3:70b", messages=messages)
    print(f"\nFinal answer:\n{final.choices[0].message.content}")
```

Run it again. This time you should see three things:
1. The model's tool call request
2. The actual scouting data from your "database"
3. A final answer that includes the REAL arm strength numbers

The answer now includes `62 mph` -- the actual stat from your database. No hallucination. The model looked it up.

---

## Step 6: Add More Tools

One tool is useful. Three tools are powerful. Let's give the model a full toolkit. Create a new file:

```python
# 15-agents-and-tool-use/ex2_multi_tool.py
"""Multiple tools -- the model picks which ones it needs."""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Three "databases"
PLAYERS = {
    "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
    "RB-201": "Explosive runner with 4.38 40-yard dash. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup.",
    "WR-301": "Crisp route runner with elite separation. Full route tree, slot and outside. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage at the line.",
}

COMBINE = {
    "QB-101": "40-yard dash: 4.71s. Vertical: 32in. Broad jump: 9'4\". Bench press: 18 reps. Hand size: 10.25in.",
    "RB-201": "40-yard dash: 4.38s. Vertical: 39in. Broad jump: 10'8\". Bench press: 22 reps. 3-cone: 6.89s.",
    "WR-301": "40-yard dash: 4.42s. Vertical: 38in. Broad jump: 10'4\". Bench press: 14 reps. 3-cone: 6.92s.",
}

DRAFT_HISTORY = {
    "qb_round1_2020-2024": "Notable 1st-round QBs: Burrow (1.01, 2020), Lawrence (1.01, 2021), Stroud (1.02, 2023), Daniels (1.02, 2024).",
    "qb_late_2020-2024": "Notable late-round QBs: Purdy (7.262, 2022). Rare to find starters after Round 4.",
    "wr_round1_2020-2024": "Notable 1st-round WRs: Lamb (1.17, 2020), Chase (1.05, 2021), Wilson (1.10, 2022), Harrison (1.04, 2024).",
}

def lookup_player_stats(player_id: str) -> str:
    return PLAYERS.get(player_id.upper(), f"Player '{player_id}' not found.")

def get_combine_results(player_id: str) -> str:
    return COMBINE.get(player_id.upper(), f"No combine data for '{player_id}'.")

def search_draft_history(query: str) -> str:
    return DRAFT_HISTORY.get(query.lower(), f"No draft history for '{query}'.")
```

Now define all three tool schemas:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_player_stats",
            "description": "Look up a football player's scouting report by their ID (e.g., QB-101, RB-201)",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "string", "description": "The player ID"}
                },
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_combine_results",
            "description": "Get NFL Combine results for a player (40 time, vertical, broad jump, bench press)",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "string", "description": "The player ID (e.g., QB-101, WR-301)"}
                },
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_draft_history",
            "description": "Search historical draft picks by position, round, and year range",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search key like 'qb_round1_2020-2024' or 'wr_round1_2020-2024'"}
                },
                "required": ["query"],
            },
        },
    },
]
```

And a helper to execute whichever tool the model calls:

```python
def execute_tool(name: str, args: dict) -> str:
    if name == "lookup_player_stats":
        return lookup_player_stats(args["player_id"])
    elif name == "get_combine_results":
        return get_combine_results(args["player_id"])
    elif name == "search_draft_history":
        return search_draft_history(args["query"])
    return "Unknown tool"
```

---

## Step 7: Ask a Complex Question

Now let's ask something that requires MULTIPLE lookups:

```python
question = "Write a scouting report for QB-101. Include his game stats, combine numbers, and how he compares to recent first-round QBs."

messages = [
    {"role": "system", "content": "You are an NFL draft analyst. Use tools to look up accurate information. Never guess stats, combine numbers, or draft history."},
    {"role": "user", "content": question},
]

print(f"Question: {question}\n")

# First call -- model requests tools
response = client.chat.completions.create(
    model="llama3.3:70b",
    messages=messages,
    tools=tools,
    temperature=0.0,
)

msg = response.choices[0].message
if msg.tool_calls:
    messages.append(msg)

    print("=== Tool Calls ===")
    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        result = execute_tool(name, args)
        print(f"  {name}({args}) -> {result[:60]}...")
        messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})

    # Final answer with all the real data
    final = client.chat.completions.create(
        model="llama3.3:70b",
        messages=messages,
        temperature=0.0,
    )
    print(f"\n=== Final Answer ===\n{final.choices[0].message.content}")
else:
    print("Model answered without tools:")
    print(msg.content)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex2_multi_tool.py
```

Watch the tool calls. The model should call multiple tools -- `lookup_player_stats`, `get_combine_results`, and `search_draft_history` -- all from a single question. It figured out which information it needed.

The final answer has real stats, actual combine numbers, and historical draft context. All from your database, not from the model's imagination.

---

## Step 8: Why JSON Schemas Are the Right Approach

The native Ollama Python library has a shortcut that lets you pass Python functions directly as tools (`tools=[lookup_player_stats]`). Ollama builds the schema from your docstrings and type hints automatically.

We're not using that shortcut — and that's intentional.

The OpenAI SDK (which is what we use throughout this course) requires explicit JSON schemas. This is actually the **industry-standard approach** because:
- It works identically with OpenAI, Azure OpenAI, Together AI, vLLM, and any other provider
- The schema is the contract between your code and the model — explicit is better than implicit
- You control exactly what description the model sees for each parameter

Here's a quick reference for the schema pattern you've already been writing:

```python
# 15-agents-and-tool-use/ex3_schema_reference.py
"""Tool schema reference -- the explicit JSON schema approach."""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

PLAYERS = {
    "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph.",
    "RB-201": "Explosive runner with 4.38 40-yard dash. 3.8 yards after contact.",
}

def lookup_player_stats(player_id: str) -> str:
    """Look up a football player's scouting report by their ID."""
    return PLAYERS.get(player_id.upper(), f"Player '{player_id}' not found.")

# Explicit JSON schema -- works with any OpenAI-compatible provider
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_player_stats",
            "description": "Look up a football player's scouting report by their ID (e.g., QB-101, RB-201)",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {
                        "type": "string",
                        "description": "The player ID to look up",
                    }
                },
                "required": ["player_id"],
            },
        },
    },
]

response = client.chat.completions.create(
    model="llama3.3:70b",
    messages=[{"role": "user", "content": "What's the arm strength for QB-101?"}],
    tools=tools,
)

print(response.choices[0].message)
```

The schema is explicit, portable, and production-ready. Write it once, use it anywhere.

---

## Step 9: Build a Simple Agent Loop

So far, you call the LLM once, it requests tools, you execute them, you call it again. That's a single round trip. An agent does this in a LOOP:

```
Think -> Act -> Observe -> Think -> Act -> Observe -> ... -> Done
```

Why? Some questions need multiple rounds of lookup. "Write a complete scouting report" might require looking up the player stats, then the combine results, then draft history, then realizing it needs a comparison with another prospect too.

Let's build it:

```python
# 15-agents-and-tool-use/ex4_agent_loop.py
"""A simple agent loop: think, act, observe, repeat."""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Same databases as before
PLAYERS = {
    "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Arm strength: 62 mph.",
    "RB-201": "Explosive runner with 4.38 40-yard dash. 3.8 yards after contact. 45 receptions out of backfield.",
    "WR-301": "Crisp route runner with elite separation. Full route tree. 4.42 speed, 38-inch vertical. 2.1% drop rate.",
}

COMBINE = {
    "QB-101": "40-yard dash: 4.71s. Vertical: 32in. Broad jump: 9'4\". Bench press: 18 reps.",
    "RB-201": "40-yard dash: 4.38s. Vertical: 39in. Broad jump: 10'8\". Bench press: 22 reps.",
    "WR-301": "40-yard dash: 4.42s. Vertical: 38in. Broad jump: 10'4\". Bench press: 14 reps.",
}

DRAFT_HISTORY = {
    "qb_round1_2020-2024": "Notable 1st-round QBs: Burrow (1.01, 2020), Lawrence (1.01, 2021), Stroud (1.02, 2023).",
    "qb_late_2020-2024": "Notable late-round QBs: Purdy (7.262, 2022). Rare to find starters after Round 4.",
}

def execute_tool(name: str, args: dict) -> str:
    if name == "lookup_player_stats":
        return PLAYERS.get(args["player_id"].upper(), f"Player '{args['player_id']}' not found.")
    elif name == "get_combine_results":
        return COMBINE.get(args["player_id"].upper(), "No combine data found.")
    elif name == "search_draft_history":
        return DRAFT_HISTORY.get(args["query"].lower(), "No draft history found.")
    return "Unknown tool"
```

Now the tools schema (same as before -- skip if you still have it):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_player_stats",
            "description": "Look up a football player's scouting report by ID",
            "parameters": {
                "type": "object",
                "properties": {"player_id": {"type": "string", "description": "The player ID"}},
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_combine_results",
            "description": "Get NFL Combine results for a player",
            "parameters": {
                "type": "object",
                "properties": {"player_id": {"type": "string", "description": "The player ID"}},
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_draft_history",
            "description": "Search historical draft picks by position, round, and year range",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Query like: qb_round1_2020-2024, qb_late_2020-2024"}},
                "required": ["query"],
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
            "You are an NFL draft analyst. "
            "Use the provided tools to look up every player stat, combine result, and draft history. "
            "NEVER guess -- always look it up. "
            "When you have all the information you need, write the final answer."
        )},
        {"role": "user", "content": question},
    ]

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        response = client.chat.completions.create(
            model="llama3.3:70b",
            messages=messages,
            tools=tools,
            temperature=0.0,
        )

        msg = response.choices[0].message

        # If there are tool calls, execute them and continue the loop
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = execute_tool(name, args)
                print(f"  Tool: {name}({args}) -> {result[:70]}")
                messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
        else:
            # No tool calls -- the model is done gathering info and giving its answer
            print(f"  Agent finished after {round_num + 1} round(s).")
            return msg.content

    return "Max rounds reached."
```

Try it:

```python
result = agent_loop(
    "Write a complete scouting report for QB-101. Include his game stats, "
    "combine numbers, and how he compares to recent first-round QBs drafted 2020-2024."
)

print(f"\n{'='*60}")
print("FINAL SCOUTING REPORT:")
print(result)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex4_agent_loop.py
```

Watch the rounds. The model calls tools, gets results, then decides if it needs more information or has enough to write the final answer. It might take 2-3 rounds.

Notice the pattern:
- Round 1: Looks up QB-101 stats and combine results
- Round 2: Searches draft history for recent first-round QBs
- Round 3: Writes the final scouting report using ALL the real data

That's the agent loop. Think, act, observe, repeat until done.

---

## Step 10: The Full Scouting Agent

Let's put it all together in a clean class. This is the culmination of everything in this module:

```python
# 15-agents-and-tool-use/ex5_scouting_agent.py
"""A complete scouting agent that gathers player data then writes draft reports."""

from openai import OpenAI
import json


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


class ScoutingAgent:
    """Agent that looks up player stats, combine results, and draft history before writing scouting reports."""

    def __init__(self, model: str = "llama3.3:70b"):
        self.model = model
        self.history = []  # Track what the agent looked up

        # Knowledge bases
        self.players = {
            "QB-101": "Pocket passer with elite accuracy. Completes 68% of passes with 2.3-second average release. Excels on intermediate routes (15-25 yards). Reads defenses pre-snap. Arm strength: 62 mph. Weakness: locks onto first read under pressure.",
            "RB-201": "Explosive runner with 4.38 40-yard dash. Exceptional vision, finds cutback lanes. 3.8 yards after contact. 45 receptions out of backfield. Weakness: pass protection and blitz pickup.",
            "WR-301": "Crisp route runner with elite separation. Full route tree, slot and outside. 4.42 speed, 38-inch vertical. 2.1% drop rate. Weakness: press coverage at the line.",
            "OL-401": "Excellent pass protection anchor. Quick lateral movement. 34-inch arms. Run blocking: 82.5/100. 2 sacks allowed in 580 snaps. Weakness: combo blocks at the second level.",
        }
        self.combine = {
            "QB-101": "40-yard dash: 4.71s. Vertical: 32in. Broad jump: 9'4\". Bench press: 18 reps. Hand size: 10.25in. Wonderlic: 34.",
            "RB-201": "40-yard dash: 4.38s. Vertical: 39in. Broad jump: 10'8\". Bench press: 22 reps. 3-cone: 6.89s.",
            "WR-301": "40-yard dash: 4.42s. Vertical: 38in. Broad jump: 10'4\". Bench press: 14 reps. 3-cone: 6.92s.",
            "OL-401": "40-yard dash: 5.18s. Vertical: 28in. Broad jump: 8'10\". Bench press: 30 reps. Arm length: 34in. Hand size: 10.5in.",
        }
        self.draft_history = {
            "qb_round1_2020-2024": "Notable 1st-round QBs: Burrow (1.01, 2020), Lawrence (1.01, 2021), Stroud (1.02, 2023), Daniels (1.02, 2024).",
            "qb_late_2020-2024": "Notable late-round QBs: Purdy (7.262, 2022). Rare to find starters after Round 4.",
            "wr_round1_2020-2024": "Notable 1st-round WRs: Lamb (1.17, 2020), Chase (1.05, 2021), Wilson (1.10, 2022), Harrison (1.04, 2024).",
            "rb_round1_2020-2024": "Notable 1st-round RBs: Robinson (1.24, 2024), Bijan (1.08, 2023). Position devalued -- fewer early picks.",
        }

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "lookup_player_stats":
            result = self.players.get(args["player_id"].upper(), f"Player '{args['player_id']}' not found.")
        elif name == "get_combine_results":
            result = self.combine.get(args["player_id"].upper(), "No combine data found.")
        elif name == "search_draft_history":
            result = self.draft_history.get(args["query"].lower(), "No draft history found.")
        else:
            result = "Unknown tool"
        self.history.append({"tool": name, "args": args, "result": result})
        return result

    def run(self, task: str, max_rounds: int = 5) -> str:
        """Run the agent loop for a given task."""

        self.history = []

        tools = [
            {"type": "function", "function": {
                "name": "lookup_player_stats",
                "description": "Look up a football player's scouting report by ID",
                "parameters": {"type": "object", "properties": {"player_id": {"type": "string", "description": "The player ID (e.g., QB-101)"}}, "required": ["player_id"]},
            }},
            {"type": "function", "function": {
                "name": "get_combine_results",
                "description": "Get NFL Combine results for a player",
                "parameters": {"type": "object", "properties": {"player_id": {"type": "string", "description": "The player ID (e.g., QB-101, WR-301)"}}, "required": ["player_id"]},
            }},
            {"type": "function", "function": {
                "name": "search_draft_history",
                "description": "Search historical draft picks by position, round, and year range",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Query like: qb_round1_2020-2024, rb_round1_2020-2024, qb_late_2020-2024"}}, "required": ["query"]},
            }},
        ]

        messages = [
            {"role": "system", "content": (
                "You are an expert NFL draft analyst. Your job is to write "
                "accurate, complete scouting reports. ALWAYS use tools to look up player stats, "
                "combine results, and draft history. Never guess or make up numbers. "
                "Include measurables, strengths, weaknesses, and historical comparisons."
            )},
            {"role": "user", "content": task},
        ]

        for round_num in range(max_rounds):
            response = client.chat.completions.create(
                model=self.model, messages=messages, tools=tools,
                temperature=0.0,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = self._execute_tool(name, args)
                    messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            else:
                return msg.content

        return "Agent reached max rounds without finishing."


# Let's use it
agent = ScoutingAgent()

task = (
    "Write a complete scouting report for QB-101. Include his game stats, "
    "combine measurables, and compare him to recent first-round QBs from 2020-2024. "
    "Also note any late-round QB success stories for context."
)

print("=== Scouting Agent ===")
print(f"Task: {task}\n")

result = agent.run(task)

print("=== Agent's Tool Usage ===")
for h in agent.history:
    print(f"  {h['tool']}({h['args']}) -> {h['result'][:60]}...")

print(f"\n{'='*60}")
print("FINAL SCOUTING REPORT:")
print(result)
```

Run it:

```bash
python3 15-agents-and-tool-use/ex5_scouting_agent.py
```

Look at the tool usage summary at the end. The agent decided what to look up, looked it up, and then wrote a scouting report grounded in real data. Every stat, combine number, and draft comparison came from your database.

---

## What You Built

Let's recap what just happened:

1. **A tool** -- a Python function the LLM can call to get real data
2. **Tool schemas** -- JSON descriptions so the model knows what tools are available
3. **The round trip** -- model requests a tool call, you execute it, feed the result back
4. **Multiple tools** -- the model picks which ones it needs based on the question
5. **The agent loop** -- think, act, observe, repeat until the model has enough info
6. **A scouting agent** -- gathers player stats, combine results, and draft history, then writes the report

The key insight: the LLM DECIDES what information it needs. You provide the tools. It calls them. The final output is grounded in real data, not hallucinated.

---

## Takeaways

1. **Tool calling lets LLMs access real data** instead of guessing -- critical for accurate scouting reports
2. **The model chooses which tools to call** based on the question and tool descriptions
3. **Good tool descriptions matter** -- the model relies on them to decide when to use each tool
4. **Agent loops enable multi-step reasoning** -- look up stats THEN write the report
5. **Your code executes the tools, not the LLM** -- the model just decides what to call

## Next Up

Agents are powerful but need guardrails. What if the model hallucinates a player stat? What if it generates a bogus scouting grade? Module 16 covers guardrails and safety -- catching bad output before it reaches the front office.
