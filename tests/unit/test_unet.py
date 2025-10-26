import pytest

torch = pytest.importorskip("torch")

from web_stable_diffusion.models.unet_2d_condition import TVMUNet2DConditionModel


def test_unet_forward_preserves_shape():
    model = TVMUNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=4,
        device="cpu",
    )

    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.tensor([10.0])
    encoder_hidden_states = torch.randn(1, 4, 16)

    with torch.no_grad():
        output = model(sample, timestep, encoder_hidden_states)

    assert output.shape == sample.shape
    assert torch.isfinite(output).all()
