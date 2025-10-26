"""Integration and reasoning tests for the omni-modal engine."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_stable_diffusion.models.miniturbo_omnimodal import (
    Application,
    Const,
    Lambda,
    MetaLogic,
    OmniModalMiniturbo,
    Var,
)


PROMPT = "A calm lake at dawn"


def test_lambda_reduction_performs_beta_reduction() -> None:
    meta = MetaLogic()
    increment = Lambda("x", Application(Const(lambda y: y + 1), Var("x")))
    expression = Application(increment, Const(5))
    assert meta.lambda_reduce(expression) == 6


def test_deduction_and_proof_layers_work_together() -> None:
    meta = MetaLogic()
    knowledge = {"theme:night": ["token:night"], "mood:calm": ["token:calm"]}
    facts = {"token:night", "token:calm"}
    closure = meta.deduce(knowledge, facts)
    assert "theme:night" in closure
    assert meta.prove("mood:calm", knowledge, facts)


def test_engine_generates_deterministic_modalities() -> None:
    engine = OmniModalMiniturbo(diffusion_steps=4)
    image_first = engine.generate_image(PROMPT, resolution=32)
    image_second = engine.generate_image(PROMPT, resolution=32)
    np.testing.assert_allclose(image_first.to_numpy(), image_second.to_numpy(), rtol=1e-6, atol=1e-6)
    assert image_first.metadata["shape"] == (32, 32, 3)


def test_volume_and_audio_shapes_are_correct() -> None:
    engine = OmniModalMiniturbo()
    volume = engine.generate_volume(PROMPT, size=24)
    assert volume.to_numpy().shape == (24, 24, 24)
    audio = engine.generate_audio(PROMPT, length=1024, sample_rate=8000)
    assert audio.to_numpy().shape == (1024,)
    assert audio.metadata["sample_rate"] == 8000


def test_video_generation_respects_shape_and_metadata() -> None:
    engine = OmniModalMiniturbo()
    frames = 6
    video = engine.generate_video(PROMPT, frames=frames, resolution=48, fps=24)
    assert video.metadata["fps"] == 24
    assert video.to_numpy().shape == (frames, 48, 48, 3)


def test_parallel_bundle_includes_all_modalities() -> None:
    engine = OmniModalMiniturbo()
    bundle = engine.generate_bundle(PROMPT, resolution=32, frames=4, audio_length=512, volume_size=20)
    assert set(bundle.keys()) == {"audio", "image", "volume", "video"}
    for result in bundle.values():
        assert isinstance(result.metadata, dict)
