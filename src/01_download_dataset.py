import json
from collections import Counter
from pathlib import Path

data_dir = Path("pubmedqa_official/data")

json_files = list(data_dir.glob("*.json"))

print("JSON files:")
for file_path in json_files:
    print(file_path)

for file_path in json_files:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"\nFile: {file_path}")
    print("Number of samples:", len(data))

    first_id = next(iter(data))
    print("First PMID:", first_id)
    print("First sample:")
    print(json.dumps(data[first_id], indent=2, ensure_ascii=False))

    labels = [
        sample["final_decision"]
        for sample in data.values()
        if "final_decision" in sample
    ]

    print("Label distribution:", Counter(labels))