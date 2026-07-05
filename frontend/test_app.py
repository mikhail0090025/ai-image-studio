import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app  # поправь если файл называется иначе

client = TestClient(app)


# =========================
# 1. PAGE TESTS
# =========================

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_home_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_editor_page():
    resp = client.get("/editor")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# =========================
# 2. STABLE DUMMY IMAGE
# =========================

def create_dummy_image():
    """
    Стабильная картинка (не random), чтобы тесты не флапали.
    """
    img = Image.new("RGB", (512, 512), color=(120, 120, 120))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return buf


# =========================
# 3. MOCK AI SERVICE
# =========================

@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.status_code = 200
            self._content = create_dummy_image().getvalue()

        def iter_content(self, chunk_size=1024):
            yield self._content

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)


# =========================
# 4. REMOVE OBJECT (SDXL + LAMA)
# =========================

@pytest.mark.parametrize("engine", ["sdxl", "lama"])
def test_remove_object_engines(engine):
    img = create_dummy_image()

    files = {
        "image": ("test.png", img, "image/png")
    }

    data = {
        "prompt": "remove object",
        "box": "[100,100,200,200]",
        "engine": engine,
        "steps": "1"
    }

    resp = client.post("/api/remove-object", files=files, data=data)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0


# =========================
# 5. EDIT OBJECT (SMOKE ONLY)
# =========================

def test_edit_object_smoke():
    img = create_dummy_image()

    files = {
        "image": ("test.png", img, "image/png")
    }

    data = {
        "edit_prompt": "make it blue",
        "box": "[100,100,200,200]",
        "steps": "1",
        "guidance_scale": "7.5",
        "strength": "1.0"
    }

    resp = client.post("/api/edit-object", files=files, data=data)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0