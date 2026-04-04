# """OpenEnv-compatible FastAPI application for the reliability environment."""

# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import Any

# from openenv.core.env_server.http_server import create_app

# from openenv_models import ReliabilityAction, ReliabilityObservation
# from server.environment import LiveSystemReliabilityEnvironment


# def _web_enabled() -> bool:
#     return os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"}


# def _build_env() -> LiveSystemReliabilityEnvironment:
#     return LiveSystemReliabilityEnvironment()


# def _configure_readme_path() -> None:
#     # OpenEnv's web UI looks at ENV_README_PATH when README is not in /app/README.md.
#     if os.getenv("ENV_README_PATH"):
#         return
#     readme_path = Path(__file__).resolve().parents[1] / "README.md"
#     if readme_path.exists():
#         os.environ["ENV_README_PATH"] = str(readme_path)


# _configure_readme_path()

# app = create_app(
#     _build_env,
#     ReliabilityAction,
#     ReliabilityObservation,
#     env_name="adaptive-rl-reliability",
# )


# if not _web_enabled():

#     @app.get("/", tags=["Environment Info"])
#     def root() -> dict[str, Any]:
#         return {
#             "name": "adaptive-rl-reliability",
#             "status": "ok",
#             "docs": "/docs",
#             "health": "/health",
#             "web_interface_enabled": False,
#         }


# def main(host: str = "0.0.0.0", port: int | None = None) -> None:
#     import uvicorn

#     resolved_port = port if port is not None else int(os.getenv("PORT", "8000"))
#     uvicorn.run(app, host=host, port=resolved_port)


# if __name__ == "__main__":
#     main()

"""OpenEnv-compatible FastAPI application for the reliability environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

from openenv_models import ReliabilityAction, ReliabilityObservation
from server.environment import LiveSystemReliabilityEnvironment


def _web_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"}


def _build_env() -> LiveSystemReliabilityEnvironment:
    return LiveSystemReliabilityEnvironment()


def _configure_readme_path() -> None:
    # OpenEnv's web UI looks at ENV_README_PATH when README is not in /app/README.md.
    if os.getenv("ENV_README_PATH"):
        return
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if readme_path.exists():
        os.environ["ENV_README_PATH"] = str(readme_path)


_configure_readme_path()

# Create FastAPI app
app: FastAPI = create_app(
    _build_env,
    ReliabilityAction,
    ReliabilityObservation,
    env_name="adaptive-rl-reliability",
)


# Root route - serve README if available, otherwise show API info
@app.get("/", tags=["Health"], response_class=HTMLResponse)
def root() -> str:
    # Try to serve README as HTML for local development
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if readme_path.exists():
        try:
            with open(readme_path, encoding="utf-8") as f:
                md_content = f.read()
            
            # Try markdown conversion if available
            try:
                import markdown
                html_content = markdown.markdown(md_content)
            except ImportError:
                # Fallback: basic HTML escaping
                import html as html_module
                html_content = f"<pre>{html_module.escape(md_content)}</pre>"
            
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Adaptive RL Reliability</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #0066cc; }}
                    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: "Courier New", monospace; }}
                    pre {{ background: #f5f5f5; padding: 12px; border-radius: 5px; overflow-x: auto; font-size: 0.9em; }}
                    a {{ color: #0066cc; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background: #f5f5f5; }}
                    .api-links {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                {html_content}
                <div class="api-links">
                    <h3>API Documentation</h3>
                    <ul>
                        <li><a href="/docs">Swagger UI</a> - Interactive API documentation</li>
                        <li><a href="/health">Health check</a></li>
                        <li><a href="/openapi.json">OpenAPI schema</a></li>
                    </ul>
                </div>
            </body>
            </html>
            """
        except Exception:
            pass
    
    # Fallback to basic HTML
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Adaptive RL Reliability</title>
        <style>
            body {{ font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #0066cc; }}
            a {{ color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Adaptive RL Reliability</h1>
        <p>OpenEnv-compatible environment for live-system autoscaling.</p>
        <h3>API Documentation</h3>
        <ul>
            <li><a href="/docs">Swagger UI</a> - Interactive API documentation</li>
            <li><a href="/health">Health check</a></li>
            <li><a href="/openapi.json">OpenAPI schema</a></li>
        </ul>
    </body>
    </html>
    """


def main() -> None:
    import uvicorn

    # Default to 8000 for local development, respect HF Spaces PORT env var
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "server.app:app",  # important for reload/import safety
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    main()