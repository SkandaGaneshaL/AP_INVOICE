import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_active_backend_does_not_load_gepa_modules():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import app.main; print(any(n == 'gepa' or n.startswith('gepa.') for n in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False"


def test_gepa_job_endpoint_is_explicitly_disabled():
    from app.main import app

    response = TestClient(app).get("/v1/extraction-rules/gepa-jobs/not-active")
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "GEPA_DISABLED"


def test_streamlit_contains_only_normal_candidate_selection():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert 'options.append("Normal OCI")' in source
    assert 'options.append("GEPA")' not in source
    assert "GEPA Optimized" not in source
