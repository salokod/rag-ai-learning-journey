import ollama

system_prompt = """You are a senior technical writer at an ISO 9001-certified manufacturing facility.
You write task descriptions following these rules:
- Active voice, 8th-grade reading level
- Include specific tools in parentheses
- Reference form numbers and specifications
- Include safety requirements when applicable
- 50-100 words per description
- End with a documentation or quality verification step"""

user_prompt = """Here are examples of correct task descriptions:

Task: Verify torque on fastener assembly
Description: Using a calibrated torque wrench (accuracy +/-2%), verify all fasteners on Assembly #4200 meet specification MT-302 requirements. Check each fastener in sequence per the torque map diagram. Record actual torque values on Form QC-110. Any fastener outside the 25-30 Nm range must be flagged and reported to the shift supervisor before proceeding.

Task: Clean CNC machine coolant reservoir
Description: Drain coolant reservoir completely using the designated waste container (yellow, labeled "Used Coolant"). Remove debris from the screen filter and inspect for damage. Flush reservoir with clean water, then refill with Type III coolant to the MAX line. Log the coolant change on the machine maintenance card and initial the daily checklist.

Now write a description for:
Task: Replace worn conveyor belt rollers"""

response = ollama.chat(
    model="gemma3:12b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    options={"temperature": 0.1, "repeat_penalty": 1.2},
)

print(response["message"]["content"])