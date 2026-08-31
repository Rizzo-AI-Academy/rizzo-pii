"""Document authenticity verification via the Stipple API (optional companion).

rizzo-pii anonymizes what a document *says*. This module answers a different,
complementary question: **can the document itself be trusted?** A tampered
contract or an AI-generated fake act anonymizes just as cleanly as a real
one — the PII map would be perfectly reversible while the underlying
document is fraudulent. Attaching a forensic authenticity warrant to the
analysis gives the professional (lawyer, accountant, notary) a trust signal
alongside the anonymization report.

Stipple (https://www.stipple.sh) is a hosted document-forensics API with a
free anonymous tier (no API key required; set STIPPLE_API_KEY for your own
metering). All functions are best-effort: failures return None and never
break the anonymization workflow.

Enable per request with ``verify=1`` (form) or ``"verify": true`` (JSON) on
``/analyze``, or globally with ``RIZZO_VERIFY_ENABLED=1``.
"""

import json
import os
import uuid
import urllib.request
from pathlib import Path
from typing import Optional

STIPPLE_BASE_URL = os.getenv("RIZZO_STIPPLE_BASE_URL", "https://www.stipple.sh")
_REQUEST_TIMEOUT = 300  # seconds


def enabled() -> bool:
    """Whether the verification add-on should run."""
    return os.getenv("RIZZO_VERIFY_ENABLED", "").lower() in ("1", "true", "yes")


def _headers() -> dict:
    headers = {
        "User-Agent": "rizzo-pii-stipple/1.0",
        "Accept": "application/json",
    }
    api_key = os.getenv("STIPPLE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post_file(endpoint: str, file_path: Path) -> Optional[dict]:
    """POST a document as multipart to a Stipple endpoint. Best-effort."""
    try:
        boundary = "----rizzo-stipple" + uuid.uuid4().hex
        with open(file_path, "rb") as f:
            content = f.read()
        body = b"".join(
            [
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{file_path.name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            STIPPLE_BASE_URL + endpoint,
            data=body,
            method="POST",
            headers={
                **_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - verification is best-effort by design
        return None


def verification_block(file_path: Optional[Path], force: bool = False) -> Optional[dict]:
    """Build the document_verification block for an uploaded file.

    Returns None when there is no file (text-only analysis), the API is
    unreachable, or verification is disabled — callers simply omit the block.
    ``force=True`` runs the check even without the global flag (per-request
    opt-in).
    """
    if file_path is None:
        return None
    if not enabled() and not force:
        return None
    block: dict = {}
    warrant = _post_file("/v1/warrants", file_path)
    if warrant:
        block["authenticity"] = {
            "warrant_id": warrant.get("warrant_id"),
            "risk_band": warrant.get("risk_band"),
            "risk_score": warrant.get("risk_score"),
            "inspection_quality": warrant.get("inspection_quality"),
            "recommended_action": warrant.get("recommended_action"),
            "summary": warrant.get("summary"),
        }
    ai = _post_file("/v1/detect-ai-text", file_path)
    if ai:
        block["ai_text"] = (
            {"applicable": False}
            if ai.get("applicable") is False
            else {
                "applicable": True,
                "probability": ai.get("probability"),
                "lean": ai.get("lean"),
                "tells": ai.get("tells"),
            }
        )
    return block or None
