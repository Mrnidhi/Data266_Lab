"""DATA266 Lab 1. Smoke outputs are diagnostic, not final results."""

from .common import TASK_DIRECTORIES, project_root

# Keep the lab1 imports stable while each implementation lives with its author.
__path__.extend(str(project_root() / folder / "srinidhi" / "src")
                for folder in TASK_DIRECTORIES.values())

__version__ = "0.1.0"
