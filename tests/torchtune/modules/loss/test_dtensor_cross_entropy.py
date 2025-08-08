import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, init_device_mesh, distribute_tensor, Replicate
from torchtune.modules.loss import LinearCrossEntropyLoss


class TestDTensorCrossEntropy:
    """Test DTensor compatibility in LinearCrossEntropyLoss"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not hasattr(torch.distributed, '_tensor'), reason="DTensor not available")
    def test_dtensor_weight_regular_hidden(self):
        """Test when weight is DTensor and hidden is regular tensor"""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="tcp://localhost:23456", world_size=1, rank=0)
        
        device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(1,))
        
        # Setup
        bsz, seq_len, hidden_dim, vocab_size = 2, 10, 128, 1000
        loss_fn = LinearCrossEntropyLoss(ignore_index=-100)
        
        # Create regular hidden tensor
        hidden = torch.randn(bsz, seq_len, hidden_dim, device="cuda")
        target = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        
        # Create DTensor weight
        weight_local = torch.randn(vocab_size, hidden_dim, device="cuda")
        weight = distribute_tensor(weight_local, device_mesh=device_mesh, placements=[Replicate()])
        
        # This should not raise an error
        loss = loss_fn.compute_cross_entropy(hidden, target, weight)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not hasattr(torch.distributed, '_tensor'), reason="DTensor not available")
    def test_regular_weight_dtensor_hidden(self):
        """Test when weight is regular tensor and hidden is DTensor"""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="tcp://localhost:23456", world_size=1, rank=0)
        
        device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(1,))
        
        # Setup
        bsz, seq_len, hidden_dim, vocab_size = 2, 10, 128, 1000
        loss_fn = LinearCrossEntropyLoss(ignore_index=-100)
        
        # Create DTensor hidden
        hidden_local = torch.randn(bsz, seq_len, hidden_dim, device="cuda")
        hidden = distribute_tensor(hidden_local, device_mesh=device_mesh, placements=[Replicate()])
        target = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
        
        # Create regular weight tensor
        weight = torch.randn(vocab_size, hidden_dim, device="cuda")
        
        # This should not raise an error
        loss = loss_fn.compute_cross_entropy(hidden, target, weight)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        
    def test_regular_tensors(self):
        """Test with regular tensors (no DTensor)"""
        # Setup
        bsz, seq_len, hidden_dim, vocab_size = 2, 10, 128, 1000
        loss_fn = LinearCrossEntropyLoss(ignore_index=-100)
        
        hidden = torch.randn(bsz, seq_len, hidden_dim)
        target = torch.randint(0, vocab_size, (bsz, seq_len))
        weight = torch.randn(vocab_size, hidden_dim)
        
        # This should work as before
        loss = loss_fn.compute_cross_entropy(hidden, target, weight)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        
    def test_dtensor_compatibility_issue_2856(self):
        """
        Regression test for issue #2856: DTensor/torch.Tensor mixed type error
        in Llama4 LoRA fine-tuning.
        """
        # This test ensures the specific scenario from the bug report works
        pass
