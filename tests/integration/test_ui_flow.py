import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pytest_playwright", reason="pytest-playwright plugin is required for browser automation tests")

pytestmark = pytest.mark.playwright


@pytest.fixture
def mock_site(tmp_path: Path) -> Path:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "mock_site"
    shutil.copytree(fixtures_dir, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_ui_generate_flow(page, mock_site: Path):
    page.route("**/huggingface.co/**", lambda route: route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps({"mock": "tokenizer"}),
    ))

    page.goto(f"file://{mock_site / 'index.html'}")

    page.wait_for_selector("#modelId")
    page.select_option("#modelId", "Stable-Diffusion-1.5")
    page.fill("#inputPrompt", "galaxy robot")
    page.fill("#negativePrompt", "blurry")
    page.select_option("#schedulerId", "1")

    page.wait_for_function(
        "() => typeof window.tvmjsGlobalEnv.asyncOnRPCServerLoad === 'function'"
    )
    page.wait_for_function("() => window.__tokenizerReady === true")

    tokenizer_info = page.evaluate(
        """
        async () => {
            await window.tvmjsGlobalEnv.asyncOnRPCServerLoad();
            const tokenizer = await window.tvmjsGlobalEnv.getTokenizer('test/model');
            return {
                initCount: window.__tokenizerInitCount,
                payload: window.__lastTokenizerPayload
            };
        }
        """
    )

    assert tokenizer_info["initCount"] == 1
    assert "mock" in tokenizer_info["payload"]

    page.click("#generate")
    page.wait_for_function(
        "() => document.querySelector('#canvas').dataset.lastRender !== undefined"
    )

    render_state = page.eval_on_selector("#canvas", "el => el.dataset.lastRender")
    render_payload = json.loads(render_state)

    assert render_payload["prompt"] == "galaxy robot"
    assert render_payload["negative"] == "blurry"
    assert render_payload["scheduler"] == "1"

    log_text = page.inner_text("#log")
    assert "Generated with galaxy robot" in log_text

    progress_value = page.eval_on_selector(
        "#progress-tracker-progress", "el => el.value"
    )
    assert progress_value == 100


def test_ui_streaming_visuals(page, mock_site: Path):
    page.route("**/huggingface.co/**", lambda route: route.fulfill(status=200, content_type="application/json", body="{}"))
    page.goto(f"file://{mock_site / 'index.html'}")

    page.click("#generate")
    page.wait_for_function(
        "() => Array.isArray(window.__streamFrames) && window.__streamFrames.length === 3"
    )

    frames = page.evaluate("() => window.__streamFrames")
    assert [frame["stage"] for frame in frames] == ["seed", "denoise", "final"]
    assert frames[-1]["progress"] == pytest.approx(1.0)

    last_stream = page.eval_on_selector("#canvas", "el => el.dataset.lastStream")
    assert json.loads(last_stream)["stage"] == "final"


def test_ui_theme_toggle_updates_root(page, mock_site: Path):
    page.route("**/huggingface.co/**", lambda route: route.fulfill(status=200, content_type="application/json", body="{}"))
    page.goto(f"file://{mock_site / 'index.html'}")

    def current_theme() -> str:
        return page.evaluate("() => document.documentElement.dataset.theme")

    assert current_theme() == "light"

    page.click("#theme-dark")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
    assert page.get_attribute("#theme-dark", "aria-pressed") == "true"
    assert page.get_attribute("#theme-light", "aria-pressed") == "false"

    page.click("#theme-custom")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'custom'")
    history = page.evaluate("() => window.__themeChanges.map(entry => entry.theme)")
    assert history[:3] == ["light", "dark", "custom"]


def test_ui_accessibility_announcements(page, mock_site: Path):
    page.route("**/huggingface.co/**", lambda route: route.fulfill(status=200, content_type="application/json", body="{}"))
    page.goto(f"file://{mock_site / 'index.html'}")

    assert page.get_attribute("#log", "role") == "log"
    assert page.get_attribute("#log", "aria-live") == "polite"

    page.click("#generate")
    page.wait_for_function("() => (window.__logEntries || []).length > 0")

    logs = page.evaluate("() => window.__logEntries")
    assert any("Generated with" in entry for entry in logs)

    aria_value = page.eval_on_selector(
        "#progress-tracker-progress", "el => el.getAttribute('aria-valuenow')"
    )
    assert aria_value == "100"

    assert page.get_by_role("img", name="Generation preview").is_visible()
