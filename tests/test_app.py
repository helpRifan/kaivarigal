import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hwrobot.app.server import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        c.put("/api/settings", json={"sim_speedup": 1e6})
        yield c


def new_page(client, template="ruled-8"):
    r = client.post("/api/pages/from-template", json={"template_id": template})
    assert r.status_code == 200
    return r.json()


def wait_done(client, job_id, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "stopped") and (job["image_url"] or job["trace_only"]):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_status(client):
    s = client.get("/api/status").json()
    assert s["backend"] == "simulator" and s["model_simulated"] is True


def test_templates_and_thumbs(client):
    tpls = client.get("/api/templates").json()
    assert {t["id"] for t in tpls} >= {"ruled-8", "lab-record"}
    r = client.get(tpls[0]["thumb_url"])
    assert r.headers["content-type"] == "image/png"


def test_page_and_image(client):
    page = new_page(client)
    assert len(page["lines"]) > 20 and page["line_spacing_mm"] == pytest.approx(8)
    r = client.get(page["image_url"])
    assert Image.open(io.BytesIO(r.content)).size == (840, 1188)


def test_photo_upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, format="JPEG")
    r = client.post("/api/pages/from-photo", files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["simulated"] is True
    bad = client.post("/api/pages/from-photo", files={"file": ("p.txt", b"nope", "text/plain")})
    assert bad.status_code == 400


def test_preview(client):
    page = new_page(client)
    r = client.post("/api/preview", json={"page_id": page["id"], "assignments": {"L1": "hello " * 30}})
    body = r.json()
    assert r.status_code == 200
    assert len(body["lines"]) >= 2 and body["lines"][0]["paths"][0].startswith("M")
    assert body["ink_mm"] > 0


def test_preview_errors(client):
    assert client.post("/api/preview", json={"page_id": "missing"}).status_code == 404
    page = new_page(client)
    r = client.post("/api/preview", json={"page_id": page["id"], "assignments": {"X9": "hi"}})
    assert r.status_code == 400
    r = client.post("/api/preview", json={"page_id": page["id"], "style": {"slant_deg": 90}})
    assert r.status_code == 422


def test_job_runs_and_lands_in_history(client):
    page = new_page(client)
    job = client.post("/api/jobs", json={"page_id": page["id"], "assignments": {"L1": "write me"}}).json()
    done = wait_done(client, job["id"])
    assert done["state"] == "done" and done["progress"] == 1.0
    assert client.get(done["image_url"]).status_code == 200
    history = client.get("/api/jobs").json()
    assert history[0]["id"] == job["id"]


def test_empty_job_rejected(client):
    page = new_page(client)
    assert client.post("/api/jobs", json={"page_id": page["id"]}).status_code == 400


def test_pause_resume_stop(client):
    client.put("/api/settings", json={"sim_speedup": 1})
    page = new_page(client)
    job = client.post("/api/jobs", json={"page_id": page["id"], "assignments": {"L1": "slow " * 20}}).json()
    jid = job["id"]
    assert client.post(f"/api/jobs/{jid}/pause").json()["state"] == "paused"
    assert client.post(f"/api/jobs/{jid}/pause").status_code == 409
    # second job blocked while one is active
    blocked = client.post("/api/jobs", json={"page_id": page["id"], "assignments": {"L1": "x"}})
    assert blocked.status_code == 409
    assert client.post(f"/api/jobs/{jid}/resume").json()["state"] == "running"
    assert client.post(f"/api/jobs/{jid}/stop").json()["state"] == "stopped"
    assert wait_done(client, jid)["state"] == "stopped"
    assert client.post(f"/api/jobs/{jid}/stop").status_code == 409


def test_trace_job_not_in_history(client):
    page = new_page(client)
    job = client.post("/api/jobs", json={"page_id": page["id"], "assignments": {"L1": "hi"},
                                         "trace_only": True}).json()
    wait_done(client, job["id"])
    assert all(h["id"] != job["id"] for h in client.get("/api/jobs").json())


def test_capture_sample_saved(client, tmp_path):
    assert client.get("/api/capture/prompts").json()["samples"] == 0
    pts = [[10, 10, 0], [20, 15, 0], [30, 10, 1]]
    r = client.post("/api/capture/samples", json={"prompt": "hi", "points": pts, "times_ms": [0, 16, 32]})
    assert r.status_code == 200 and r.json()["samples"] == 1
    assert len(list((tmp_path / "samples").glob("*.json"))) == 1
    bad = client.post("/api/capture/samples", json={"prompt": "hi", "points": [[1, 1, 0]]})
    assert bad.status_code == 400


def test_calibration_flow(client):
    job = client.post("/api/calibration/grid").json()
    assert wait_done(client, job["id"])["state"] == "done"
    m = client.post("/api/calibration/measure").json()
    assert m["simulated"] is True
    s = client.post("/api/calibration/apply", json=m).json()
    assert s["calibration"]["scale"] == m["scale"]


def test_settings_validation(client):
    assert client.put("/api/settings", json={"feed_rate": 10}).status_code == 400
    assert client.put("/api/settings", json={"feed_rate": 2000}).json()["feed_rate"] == 2000
