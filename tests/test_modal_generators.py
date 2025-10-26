"""Regression tests for the OmniModal prototype generators."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_stable_diffusion.models.miniturbo_omnimodal import OmniModalMiniturbo
from web_stable_diffusion.models import modal_generators


PROMPT = "A calm lake at dawn"


def _assert_matches_reference(result, expected: np.ndarray) -> None:
    assert result.metadata["shape"] == expected.shape
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_audio_generator_regression() -> None:
    engine = OmniModalMiniturbo()
    result = engine.generate_audio(PROMPT)
    expected = modal_generators.audio_reference(PROMPT)
    _assert_matches_reference(result, expected)


def test_image_generator_regression() -> None:
    engine = OmniModalMiniturbo()
    resolution = 32
    result = engine.generate_image(PROMPT, resolution)
    expected = modal_generators.image_reference(PROMPT, resolution, resolution)
    _assert_matches_reference(result, expected)


def test_volume_generator_regression() -> None:
    engine = OmniModalMiniturbo()
    size = 12
    result = engine.generate_volume(PROMPT, size)
    expected = modal_generators.volume_reference(PROMPT, size)
    _assert_matches_reference(result, expected)


def test_video_generator_regression() -> None:
    engine = OmniModalMiniturbo()
    frames = 6
    resolution = 10
    result = engine.generate_video(PROMPT, frames=frames, resolution=resolution)
    expected = modal_generators.video_reference(PROMPT, frames, resolution, resolution)
    _assert_matches_reference(result, expected)
