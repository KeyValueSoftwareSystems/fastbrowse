from fastbrowse.cli import main
from fastbrowse.tracing import init_observability

init_observability()

__all__ = ["main"]
