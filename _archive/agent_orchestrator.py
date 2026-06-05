"""
Archived agentic orchestrator placeholder.

The previous implementation depended on a local LLM path that is no longer part
of the active RunPod/API-based service. It is intentionally disabled so archive
imports do not pull inactive model dependencies back into the project.
"""


def main() -> None:
    raise RuntimeError(
        "Archived agentic orchestrator is disabled. Use webservice.py /recommend."
    )


if __name__ == "__main__":
    main()
