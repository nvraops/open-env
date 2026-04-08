from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_RUN_PATH = Path(__file__).resolve().parent / "inference" / "run.py"
_SPEC = spec_from_file_location("openenv_inference_run", _RUN_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load inference logic from {_RUN_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

get_action = _MODULE.get_action
load_data = _MODULE.load_data

__all__ = ["get_action", "load_data"]
