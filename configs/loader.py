from pathlib import Path
import ast


def load_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        value = value.strip()

        if value:
            current[key] = ast.literal_eval(value)
        else:
            current[key] = {}
            stack.append((indent, current[key]))

    return root


def load_env_config() -> dict:
    return load_simple_yaml(Path(__file__).with_name("env_config.yaml"))
