import pytest
from peft.tuners.lora.config import LoraConfig
from transformers import AutoModel


@pytest.fixture(
    scope="session",
)
def base_model():
    """Load a lightweight model for testing."""
    base_model = AutoModel.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2",
        torchscript=True,
    )

    return base_model


@pytest.fixture(
    scope="session",
)
def lora_config():
    # Create a test LoRA configuration. Adjust parameters as needed.
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.01,
        target_modules=[
            "query",
            "key",
            "value",
        ],
    )

    return config
