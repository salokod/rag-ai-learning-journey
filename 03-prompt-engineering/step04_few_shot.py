import ollama

few_shot_prompt = """You write manufacturing task descriptions. Here are examples of the correct format:

EXAMPLE 1:
Task: Verify torque on fastener assembly
Description: Using a calibrated torque wrench (accuracy +/-2%), verify all fasteners on Assembly #4200 meet specification MT-302 requirements. Check each fastener in sequence per the torque map diagram. Record actual torque values on Form QC-110. Any fastener outside the 25-30 Nm range must be flagged and reported to the shift supervisor before proceeding.

EXAMPLE 2:
Task: Clean CNC machine coolant reservoir
Description: Drain coolant reservoir completely using the designated waste container (yellow, labeled "Used Coolant"). Remove debris from the screen filter and inspect for damage. Flush reservoir with clean water, then refill with Type III coolant to the MAX line. Log the coolant change on the machine maintenance card and initial the daily checklist.

Now write a description in the same format for:
Task: Inspect incoming raw steel plates for surface defects"""

response = ollama.chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": few_shot_prompt}],
    options={"temperature": 0.1},
)

print(response["message"]["content"])