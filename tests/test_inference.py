import numpy as np
import torch
import pytest
from app.backend.inference import transform

@pytest.mark.parametrize(
    "image,description",
    [
        (np.random.rand(400, 400), "Random 400x400 input"),
        (np.random.rand(300, 300), "Random 300x300 input"),
        (np.ones((400, 400)), "All ones"),
        (np.zeros((400, 400)), "All zeros"),
    ]
)

def test_transfrom(image: np.array, description: str):
    image = transform(image).unsqueeze(0)
    assert image.shape == (1, 1, 28, 28), description
    assert image.max() <= 1.0001 and image.min() >= 0, description

