import numpy as np
import pytest
import torch
import torch.nn as nn
from torchtune import utils
from torchtune.modules import TransformerDecoder
from torchtune.modules.early_exit_loss import (
    early_exit_loss,
    layer_ids_to_loss_scales,
    LossScaleType,
    EarlyExitCurriculumType,
    build_early_exit_curriculum,
    RotationalEarlyExitCurriculum,
    GradualEarlyExitCurriculum,
)

# Mock components for TransformerDecoder
class MockLayer(nn.Module):
    def forward(self, x, mask=None, encoder_input=None, encoder_mask=None, input_pos=None):
        return x  # Simply return the input for testing purposes

@pytest.fixture
def mock_model():
    # Create mock components
    tok_embeddings = nn.Embedding(1000, 512)  # Example vocab size and embedding dim
    layers = nn.ModuleList([MockLayer() for _ in range(12)])  # 12 mock layers
    norm = nn.LayerNorm(512)  # Example layer normalization
    output = nn.Linear(512, 1000)  # Example output layer

    # Create an instance of TransformerDecoder
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=512,
        num_heads=8,
        head_dim=64,
        norm=norm,
        output=output,
        num_layers=12,
        output_hidden_states=[0, 1, 2]  # Example layers to output hidden states
    )
    return model

@pytest.fixture
def hidden_states_dict():
    return {i: torch.randn(4, 5, 512) for i in range(3)}  # Adjusted embedding dim

@pytest.fixture
def labels():
    return torch.randint(0, 1000, (4, 5))  # Adjusted vocab size

@pytest.fixture
def loss_fn():
    return nn.CrossEntropyLoss(ignore_index=-1)

def test_early_exit_loss(mock_model, hidden_states_dict, labels, loss_fn):
    loss = early_exit_loss(mock_model, hidden_states_dict, labels, loss_fn)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_layer_ids_to_loss_scales():
    layer_ids = torch.tensor([0, 1, 2])
    n_layers = 12
    scales = layer_ids_to_loss_scales(layer_ids, n_layers, LossScaleType.SUM_L, 1.0)
    assert torch.isclose(scales.sum(), torch.tensor(1.0))

def test_build_early_exit_curriculum():
    curriculum = build_early_exit_curriculum(EarlyExitCurriculumType.ROTATIONAL, [True, False, True], 100)
    assert isinstance(curriculum, RotationalEarlyExitCurriculum)

    curriculum = build_early_exit_curriculum(EarlyExitCurriculumType.GRADUAL, [True, False, True], 100)
    assert isinstance(curriculum, GradualEarlyExitCurriculum)

def test_rotational_early_exit_curriculum():
    curriculum = RotationalEarlyExitCurriculum([True, False, True], 100)
    curriculum.step()
    expected = np.array([False, True, True])
    assert np.array_equal(curriculum.get(), expected), f"Expected {expected}, but got {curriculum.get()}"

def test_gradual_early_exit_curriculum():
    curriculum = GradualEarlyExitCurriculum([False, False, False, False], max_steps=4, percent_scale=1)
    curriculum.step()
    assert curriculum.get() == [False, False, False, False]
    curriculum.step()
    assert curriculum.get() == [False, False, False, True]
    curriculum.step()
    assert curriculum.get() == [False, False, True, True]
    curriculum.step()
    assert curriculum.get() == [False, True, True, True]
    curriculum.step()
    assert curriculum.get() == [True, True, True, True]
    curriculum.step()
    assert curriculum.get() == [True, True, True, True]

@pytest.fixture
def hidden_states_dict():
    return {i: torch.randn(4, 5, 512) for i in range(3)}  # Adjusted embedding dim

@pytest.fixture
def labels():
    return torch.randint(0, 1000, (4, 5))  # Adjusted vocab size

@pytest.fixture
def loss_fn():
    return nn.CrossEntropyLoss(ignore_index=-1)

def test_early_exit_loss_vs_manual(mock_model, hidden_states_dict, labels, loss_fn):
    # Convert to float32 for numeric equivalence
    # Calculate early exit loss using the function
    calculated_loss = early_exit_loss(mock_model, hidden_states_dict, labels, loss_fn, e_scale=1, loss_scale_type="one")
    # Manually calculate the loss for each hidden state
    total_loss = 0.0
    num_hidden_states = len(hidden_states_dict)
    for i, hidden_state in hidden_states_dict.items():
        # Compute logits for the current hidden state
        logits = mock_model.unembed(hidden_state)
        logits = logits.view(-1, logits.size(-1))  # Flatten for loss computation
        labels = labels.view(-1)
        # Compute the loss for the current hidden state
        loss = loss_fn(logits, labels)
        total_loss += loss
    # Average the losses across all hidden states
    manual_loss = total_loss / num_hidden_states
    # Compare the two losses
    assert torch.isclose(calculated_loss, manual_loss, atol=1e-6), \
        f"Calculated loss: {calculated_loss}, Manual loss: {manual_loss}"

if __name__ == "__main__":
    pytest.main()
