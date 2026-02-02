import json
import sys
def validate(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                print(f"Line: {line}")
                sys.exit(1)
            
            for k in ("instruction", "input", "output"):
                if k not in obj:
                    print(f"Error: {k} not found in line {i}")
                    sys.exit(1)

    print(f"Validation passed for {path}")

if __name__ == "__main__":
    validate(sys.argv[1])