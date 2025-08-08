# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from unittest.mock import Mock
from torchtune.modules.loss import LinearCrossEntropyLoss
from recipes.validation import validate_custom_sharding_config


class TestValidateCustomShardingConfig:
    """Unit tests for validate_custom_sharding_config"""
    
    def test_missing_output_raises_error(self):
        """Test that validation raises error when output is missing"""
        loss_fn = LinearCrossEntropyLoss()
        custom_sharded_layers = ['tok_embeddings']
        
        with pytest.raises(ValueError, match="'output' must be included"):
            validate_custom_sharding_config(loss_fn, custom_sharded_layers)
    
    def test_with_output_passes(self):
        """Test that validation passes when output is included"""
        loss_fn = LinearCrossEntropyLoss()
        custom_sharded_layers = ['tok_embeddings', 'output']
        
        # Should not raise
        validate_custom_sharding_config(loss_fn, custom_sharded_layers)
    
    def test_none_layers_passes(self):
        """Test that validation passes when custom_sharded_layers is None"""
        loss_fn = LinearCrossEntropyLoss()
        
        # Should not raise
        validate_custom_sharding_config(loss_fn, None)
    
    def test_empty_layers_passes(self):
        """Test that validation passes when custom_sharded_layers is empty"""
        loss_fn = LinearCrossEntropyLoss()
        
        # Should not raise
        validate_custom_sharding_config(loss_fn, [])
    
    def test_parallelism_disabled_skips_validation(self):
        """Test that validation is skipped when parallelism is disabled"""
        loss_fn = LinearCrossEntropyLoss()
        custom_sharded_layers = ['tok_embeddings']  # Missing output
        
        # Should not raise because parallelism_enabled=False
        validate_custom_sharding_config(
            loss_fn, 
            custom_sharded_layers,
            parallelism_enabled=False
        )
    
    def test_non_linear_ce_loss_passes(self):
        """Test that non-LinearCrossEntropyLoss doesn't require output"""
        loss_fn = Mock()  # Some other loss function
        custom_sharded_layers = ['tok_embeddings']  # Missing output
        
        # Should not raise
        validate_custom_sharding_config(loss_fn, custom_sharded_layers)
