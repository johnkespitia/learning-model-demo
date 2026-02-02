import json

SYSTEM_PROMPT = (
    "You are an AI tutor engine. Output must be bilingual (ES/EN). "
    "Always end with an ActionableData JSON block inside triple backticks. "
    "Follow the user's pedagogical rules and adapt to student_snapshot."
)

def build_prompt(exmaple: dict) -> str:
    instruction = example["instruction"]
    inp = example["input"]
    inp_json = json.dumps(inp, ensure_ascii=True)

    user = (
        f"Task: {instruction}\n"
        f"InputJSON: {inp_json}\n"
        f"Requirements:\n"
        f"- bilingual ES/EN\n"
        f"- include structured sections\n"
        f"- end with ActionableData JSON in triple backticks\n"
    )

    return f"<s>[INST] <<SYS>>{SYSTEM}<</SYS>>\n{user} [/INST]"

def build_pair(example: dict) :
    prompt = build_prompt(example)
    completion = example["output"]
    return prompt, completion