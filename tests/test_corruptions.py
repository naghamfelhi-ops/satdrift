import pytest
import torch
from satdrift.drift.corruptions import CORRUPTIONS, LEVELS


@pytest.fixture
def img():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 64, 64, generator=g)


@pytest.mark.parametrize("name", CORRUPTIONS)
@pytest.mark.parametrize("sev", LEVELS)
def test_shape_and_range(name, sev, img):
    out = CORRUPTIONS[name](img, sev)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_deterministic(name, img):
    assert torch.equal(CORRUPTIONS[name](img, 3), CORRUPTIONS[name](img, 3))


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_monotonic_distortion(name, img):
    """Plus la sévérité augmente, plus l'image s'éloigne de l'originale."""
    dists = [(CORRUPTIONS[name](img, s) - img).abs().mean().item() for s in LEVELS]
    assert dists == sorted(dists)


def test_invalid_severity(img):
    with pytest.raises(ValueError):
        CORRUPTIONS["blur"](img, 9)

def test_batch_input(img):
    batch = torch.stack([img, img])
    for name, fn in CORRUPTIONS.items():
        assert fn(batch, 3).shape == batch.shape


def test_noise_differs_between_images(img):
    """Deux images identiques dans un lot doivent recevoir un bruit différent."""
    out = CORRUPTIONS["noise"](torch.stack([img, img]), 3)
    assert not torch.equal(out[0], out[1])