"""Share the upstream test's synthetic model with the optional notarial app.

No model files are loaded and no real application preferences are consulted.
Missing Flask/PyMuPDF still allows the standard-library-only suite to run;
an unexpected failure in the notarial module is deliberately not hidden.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/app"))
import server_config


# The upstream helper mocks transformers but its fallback also imports torch.
# Supply only this inert test double, without installing/loading torch or weights.
previous_torch = sys.modules.get("torch")
fake_torch = MagicMock()
fake_torch.cuda.is_available.return_value = False
sys.modules["torch"] = fake_torch
try:
    with patch.object(server_config, "load_prefs", return_value={
            "excluded_tags": [], "mapping_enabled": True, "ocr_mode": "auto",
            "ocr_preset": "balanced", "ocr_redact_seals": True}):
        from test_ora import APP as STANDARD_APP
finally:
    if previous_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = previous_torch


APP = None
if STANDARD_APP is not None:
    import notarial_server_config

    with patch.object(notarial_server_config, "load_prefs", return_value={
            "excluded_tags": [], "mapping_enabled": False, "ocr_mode": "auto",
            "ocr_preset": "balanced", "ocr_redact_seals": False}):
        import notarial_app as APP
