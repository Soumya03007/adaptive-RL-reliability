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


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_env_catalog() -> dict:
    return load_simple_yaml(Path(__file__).with_name("env_config.yaml"))


def get_default_task_name() -> str:
    catalog = load_env_catalog()
    return catalog.get("default_task", "balanced")


def list_task_names() -> list[str]:
    catalog = load_env_catalog()
    tasks = catalog.get("tasks", {})
    return list(tasks.keys())


def load_env_config(task_name: str | None = None) -> dict:
    catalog = load_env_catalog()
    selected_task = task_name or catalog.get("default_task")

    base_config = {
        key: value
        for key, value in catalog.items()
        if key not in {"default_task", "tasks"}
    }

    if not selected_task:
        return base_config

    task_overrides = catalog.get("tasks", {})
    if selected_task not in task_overrides:
        available = ", ".join(sorted(task_overrides))
        raise KeyError(f"Unknown task '{selected_task}'. Available tasks: {available}")

    return deep_merge(base_config, task_overrides[selected_task])
