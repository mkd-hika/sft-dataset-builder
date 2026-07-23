"""Console-script entry point: `sft-dataset-builder-ui` launches the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def launch_ui() -> None:
    from streamlit.web import cli as stcli

    app_path = str(Path(__file__).parent / "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    launch_ui()
