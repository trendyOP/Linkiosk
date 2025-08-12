# auto_click.py
import json
import random
from action import action

with open("output.json", "r") as f:
    data = json.load(f)

print("읽어온 UI 개수:", len(data))

target = random.choice(data)
print("선택된 UI:", target)

action(target)
