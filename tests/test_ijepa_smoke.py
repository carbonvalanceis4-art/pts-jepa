import torch

from models.ijepa_tiny import IJEPATiny, random_target_mask


def test_ijepa_forward_and_ema_update():
    model = IJEPATiny(image_size=64, patch_size=16, dim=96, depth=2, heads=3)
    x = torch.randn(2, 3, 64, 64)
    mask = random_target_mask(2, model.context_encoder.num_patches, device=x.device)
    loss = model(x, mask)
    assert torch.isfinite(loss)
    loss.backward()
    model.update_target(0.99)
