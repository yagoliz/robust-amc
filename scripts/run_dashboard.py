#!/usr/bin/env python3
"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py
    # or
    uv run streamlit run app/main.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the app path
    app_path = Path(__file__).parent.parent / "app" / "main.py"

    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)

    # Launch streamlit
    print("Launching Robust AMC Dashboard...")
    print(f"App: {app_path}")
    print("-" * 40)

    try:
        subprocess.run(
            ["streamlit", "run", str(app_path)],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("Error: streamlit not found. Install with: uv add streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()