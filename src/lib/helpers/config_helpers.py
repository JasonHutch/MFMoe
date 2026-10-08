import sys
import tomllib
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from this file (src/helpers/...) until the directory holding data/ is found.

    Resolving from the file location instead of the working directory lets every script run
    from anywhere: python src/annotate.py, python -m src.synthetic, notebooks, cron, whatever.
    """
    here = Path(__file__).resolve().parent
    return next(p for p in [here, *here.parents] if (p / "data").is_dir())


def load_toml_config(path: str | Path) -> dict:
    """Read a TOML config file into a plain dict (stdlib reader, no extra dependency)."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def project_path(*parts: str | Path) -> Path:
    """Path under the project root, e.g. project_path("data", "synthetic-annotations")."""
    return find_project_root().joinpath(*parts)


def ensure_project_on_path() -> None:
    """Make `src.*` imports work when a script is run directly (python src/annotate.py)."""
    root = str(find_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)