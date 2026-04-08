# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Global D2MoRA (GD2MoRA): Global Many-to-Many Decomposed Low-Rank Adapter

This module implements a global version of D2MoRA where:
1. A global pool of factorized matrices (A and B) is shared across all layers
2. Each layer has its own router (layer-specific routing)
3. Efficient vectorized operations (no Python-level loops over batch/seq)
4. Global load balancing loss across all layers
"""

import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class GD2MoRAConfig(PeftConfig):
    """
    Configuration class for Global D2MoRA (GD2MoRA).
    
    Args:
        r (`int`): Rank of the low-rank decomposition
        target_modules (`Union[List[str],str]`): The names of the modules to apply GD2MoRA to
        lora_alpha (`float`): Alpha parameter for scaling
        lora_dropout (`float`): Dropout probability for GD2MoRA layers
        fan_in_fan_out (`bool`): Set to True if the layer stores weights as (fan_in, fan_out)
        num_experts_a (`int`): Number of global A matrices (down-projection experts)
        num_experts_b (`int`): Number of global B matrices (up-projection experts)
        top_k_a (`int`): Number of A experts to select per forward pass
        top_k_b (`int`): Number of B experts to select per forward pass
        load_balancing_loss_weight (`float`): Weight for the load balancing loss
        bias (`str`): Bias type for GD2MoRA. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`): List of modules apart from GD2MoRA layers to be set as trainable
    """

    r: int = field(default=8, metadata={"help": "Low-rank dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with GD2MoRA."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'"
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "GD2MoRA alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "GD2MoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    num_experts_a: int = field(default=4, metadata={"help": "Number of global A matrix experts"})
    num_experts_b: int = field(default=4, metadata={"help": "Number of global B matrix experts"})
    top_k_a: int = field(default=2, metadata={"help": "Number of A experts to use per forward pass"})
    top_k_b: int = field(default=2, metadata={"help": "Number of B experts to use per forward pass"})
    load_balancing_loss_weight: float = field(default=0.01, metadata={"help": "Weight for load balancing loss"})
    bias: str = field(default="none", metadata={"help": "Bias type for GD2MoRA. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from GD2MoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.GD2MORA


class GlobalExpertPool(nn.Module):
    """
    Global pool of factorized experts shared across all layers.
    
    This maintains:
    - A pool of A matrices (input -> hidden) - SHARED globally
    - A pool of B matrices (hidden -> output) - SHARED globally
    
    Routers are NOT stored here - each layer has its own router.
    """
    
    def __init__(self, config: GD2MoRAConfig, in_features: int, out_features: int):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.r = config.r
        self.num_experts_a = config.num_experts_a
        self.num_experts_b = config.num_experts_b
        
        # Global expert pools - SHARED across all layers
        # A experts: map from input features to rank-r space
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, self.r, bias=False) 
            for _ in range(self.num_experts_a)
        ])
        # B experts: map from rank-r space to output features
        self.experts_b = nn.ModuleList([
            nn.Linear(self.r, out_features, bias=False) 
            for _ in range(self.num_experts_b)
        ])
        
        self.scaling = config.lora_alpha / self.r if config.lora_alpha else 1.0
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize expert weights"""
        for expert_a in self.experts_a:
            nn.init.kaiming_uniform_(expert_a.weight, a=math.sqrt(5))
        for expert_b in self.experts_b:
            nn.init.zeros_(expert_b.weight)
    
    def forward(self, x: torch.Tensor, router_logits_a: torch.Tensor, router_logits_b: torch.Tensor):
        """
        Forward pass through global expert pool using layer-specific router logits.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            router_logits_a: Router logits for A experts from layer-specific router
                           Shape: (batch_size * seq_len, num_experts_a)
            router_logits_b: Router logits for B experts from layer-specific router
                           Shape: (batch_size * seq_len, num_experts_b)
                           
        Returns:
            output: Output tensor of shape (batch_size, seq_len, out_features)
            aux_loss_a: Auxiliary loss for A expert load balancing
            aux_loss_b: Auxiliary loss for B expert load balancing
        """
        batch_size, seq_len, _ = x.shape
        flat_x = x.view(-1, self.in_features)  # (batch*seq, in_features)
        
        # Select top-k A experts
        topk_logits_a, topk_indices_a = torch.topk(router_logits_a, self.config.top_k_a, dim=-1)
        weights_a = F.softmax(topk_logits_a, dim=-1)  # (batch*seq, top_k_a)
        
        # Compute outputs from ALL A experts efficiently
        # Stack all expert outputs: (batch*seq, num_experts_a, r)
        a_outputs_all = torch.stack([expert(flat_x) for expert in self.experts_a], dim=1)
        
        # Gather selected experts using advanced indexing
        # Expand indices for gathering: (batch*seq, top_k_a, 1)
        indices_expanded_a = topk_indices_a.unsqueeze(-1).expand(-1, -1, self.r)
        # Gather: (batch*seq, top_k_a, r)
        selected_a_outputs = torch.gather(a_outputs_all, 1, indices_expanded_a)
        
        # Weight and sum: (batch*seq, r)
        mid_result = (selected_a_outputs * weights_a.unsqueeze(-1)).sum(dim=1)
        
        # Compute outputs from ALL B experts efficiently
        # Stack all expert outputs: (batch*seq, num_experts_b, out_features)
        b_outputs_all = torch.stack([expert(mid_result) for expert in self.experts_b], dim=1)
        
        # Select top-k B experts
        topk_logits_b, topk_indices_b = torch.topk(router_logits_b, self.config.top_k_b, dim=-1)
        weights_b = F.softmax(topk_logits_b, dim=-1)  # (batch*seq, top_k_b)
        
        # Gather selected experts: (batch*seq, top_k_b, out_features)
        indices_expanded_b = topk_indices_b.unsqueeze(-1).expand(-1, -1, self.out_features)
        selected_b_outputs = torch.gather(b_outputs_all, 1, indices_expanded_b)
        
        # Weight and sum, then scale: (batch*seq, out_features)
        output_flat = (selected_b_outputs * weights_b.unsqueeze(-1)).sum(dim=1) * self.scaling
        
        # Reshape back to (batch, seq, out_features)
        output = output_flat.view(batch_size, seq_len, self.out_features)
        
        # Compute auxiliary losses for load balancing
        # Router probability distributions (averaged over all tokens)
        router_probs_a = F.softmax(router_logits_a, dim=-1).mean(dim=0)  # (num_experts_a,)
        router_probs_b = F.softmax(router_logits_b, dim=-1).mean(dim=0)  # (num_experts_b,)
        
        # Load balancing loss: encourage uniform routing using coefficient of variation squared
        aux_loss_a = self.num_experts_a * torch.var(router_probs_a)
        aux_loss_b = self.num_experts_b * torch.var(router_probs_b)
        
        return output, aux_loss_a, aux_loss_b


class GD2MoRALayer:
    """Mixin class for GD2MoRA layers"""
    
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class LayerRouter(nn.Module):
    """
    Layer-specific router that produces routing logits for A and B experts.
    
    Each layer has its own router to make layer-specific routing decisions.
    """
    
    def __init__(self, in_features: int, r: int, num_experts_a: int, num_experts_b: int):
        super().__init__()
        # Router A: maps input features to expert selection scores
        self.router_a = nn.Linear(in_features, num_experts_a, bias=False)
        # Router B: maps from rank-r space to expert selection scores
        self.router_b = nn.Linear(r, num_experts_b, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize router weights"""
        nn.init.kaiming_uniform_(self.router_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.router_b.weight, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor, mid_result: Optional[torch.Tensor] = None):
        """
        Compute router logits for A and B experts.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features) or (batch*seq, in_features)
            mid_result: Intermediate result after A experts (batch*seq, r), optional
            
        Returns:
            router_logits_a: Routing logits for A experts
            router_logits_b: Routing logits for B experts
        """
        # Compute router logits for A experts
        router_logits_a = self.router_a(x.float())
        
        # Compute router logits for B experts (requires mid_result)
        if mid_result is not None:
            router_logits_b = self.router_b(mid_result.float())
        else:
            # Return None for router_logits_b - will be computed later
            router_logits_b = None
        
        return router_logits_a, router_logits_b


class GD2MoRALinear(nn.Linear, GD2MoRALayer):
    """Linear layer with Global D2MoRA adaptation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        global_pool: Optional[GlobalExpertPool] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GD2MoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        self.global_pool = global_pool
        
        if r > 0:
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            
            # Create layer-specific router
            self.router = LayerRouter(
                in_features=in_features,
                r=r,
                num_experts_a=global_pool.num_experts_a,
                num_experts_b=global_pool.num_experts_b
            )
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # Router and global pool have their own initialization
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, 'router'):
            self.router.train(mode)
        if self.global_pool is not None:
            self.global_pool.train(mode)
        
        if not mode and self.merge_weights and not self.merged:
            # Merging is complex for global MoE, skip it
            pass
        elif self.merge_weights and self.merged:
            self.merged = False
    
    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        
        if self.disable_adapters or not hasattr(self, 'router'):
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            # Store zero aux losses for consistency
            self._last_aux_loss_a = torch.tensor(0.0, device=x.device)
            self._last_aux_loss_b = torch.tensor(0.0, device=x.device)
            return result
        
        # Base linear transformation
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        # Add GD2MoRA contribution
        if self.r > 0 and self.global_pool is not None:
            x_dropped = self.lora_dropout(x.to(next(self.global_pool.parameters()).dtype))
            batch_size, seq_len, _ = x_dropped.shape
            flat_x = x_dropped.view(-1, self.in_features)
            
            # First, compute A expert outputs and get router logits for A
            router_logits_a, _ = self.router(flat_x)
            
            # Select top-k A experts
            topk_logits_a, topk_indices_a = torch.topk(router_logits_a, self.global_pool.config.top_k_a, dim=-1)
            weights_a = F.softmax(topk_logits_a, dim=-1)
            
            # Compute outputs from ALL A experts efficiently
            a_outputs_all = torch.stack([expert(flat_x) for expert in self.global_pool.experts_a], dim=1)
            indices_expanded_a = topk_indices_a.unsqueeze(-1).expand(-1, -1, self.global_pool.r)
            selected_a_outputs = torch.gather(a_outputs_all, 1, indices_expanded_a)
            mid_result = (selected_a_outputs * weights_a.unsqueeze(-1)).sum(dim=1)
            
            # Now compute router logits for B using mid_result
            _, router_logits_b = self.router(flat_x, mid_result)
            
            # Select top-k B experts
            topk_logits_b, topk_indices_b = torch.topk(router_logits_b, self.global_pool.config.top_k_b, dim=-1)
            weights_b = F.softmax(topk_logits_b, dim=-1)
            
            # Compute outputs from ALL B experts efficiently
            b_outputs_all = torch.stack([expert(mid_result) for expert in self.global_pool.experts_b], dim=1)
            indices_expanded_b = topk_indices_b.unsqueeze(-1).expand(-1, -1, self.out_features)
            selected_b_outputs = torch.gather(b_outputs_all, 1, indices_expanded_b)
            moe_output_flat = (selected_b_outputs * weights_b.unsqueeze(-1)).sum(dim=1) * self.scaling
            
            # Reshape back
            moe_output = moe_output_flat.view(batch_size, seq_len, self.out_features)
            result = result + moe_output
            
            # Compute auxiliary losses for load balancing and store them
            router_probs_a = F.softmax(router_logits_a, dim=-1).mean(dim=0)
            router_probs_b = F.softmax(router_logits_b, dim=-1).mean(dim=0)
            self._last_aux_loss_a = self.global_pool.num_experts_a * torch.var(router_probs_a)
            self._last_aux_loss_b = self.global_pool.num_experts_b * torch.var(router_probs_b)
            
            if result.dtype != previous_dtype:
                result = result.to(previous_dtype)
            
            return result
        
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        
        # Store zero aux losses for consistency
        self._last_aux_loss_a = torch.tensor(0.0, device=x.device)
        self._last_aux_loss_b = torch.tensor(0.0, device=x.device)
        return result


class GD2MoRAModel(torch.nn.Module):
    """
    Global D2MoRA model that wraps a pretrained transformer model.
    
    This creates a global expert pool that is shared across all adapted layers,
    reducing total parameters while maintaining flexibility.
    """
    
    def __init__(self, config: GD2MoRAConfig, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self.global_pool_a = None
        self.global_pool_b = None
        self._find_and_replace()
        mark_only_gd2mora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward
    
    def _find_and_replace(self):
        """Find target modules and replace them with GD2MoRA layers"""
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use GD2MoRA with 8-bit quantization, please install the `bitsandbytes` package."
            )
        
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        
        # Get dimensions from the first target module to create global pool
        first_module_dims = None
        key_list = [key for key, _ in self.model.named_modules()]
        
        # First pass: find dimensions
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            
            if target_module_found:
                parent, target, target_name = self._get_submodules(key)
                if isinstance(target, torch.nn.Linear):
                    first_module_dims = (target.in_features, target.out_features)
                    break
        
        if first_module_dims is None:
            raise ValueError("No suitable target modules found for GD2MoRA")
        
        # Create global expert pools
        # Note: For simplicity, we assume same dimensions across target modules
        # In practice, you might need separate pools for different dimensions
        self.global_pool = GlobalExpertPool(
            self.peft_config,
            first_module_dims[0],
            first_module_dims[1]
        )
        
        # If using shared routers, they're already in the global pool
        # Second pass: replace modules
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                
                parent, target, target_name = self._get_submodules(key)
                
                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = GD2MoRALinear(
                        target.in_features,
                        target.out_features,
                        bias=target.bias is not None,
                        global_pool=self.global_pool,
                        **kwargs
                    )
                    self._replace_module(parent, target_name, new_module, target)
        
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model."
            )
    
    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name
    
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
        
        for name, module in new_module.named_modules():
            if "lora_" in name or "expert" in name or "router" in name:
                module.to(old_module.weight.device)
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    @property
    def modules_to_save(self):
        return None
    
    def get_aux_loss(self):
        """
        Get the accumulated auxiliary load balancing loss from all layers.
        
        This method should be called after each forward pass to retrieve
        the global load balancing loss that should be added to the main loss.
        
        Returns:
            Total auxiliary loss across all layers and both A/B experts
        """
        total_aux_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for module in self.model.modules():
            if isinstance(module, GD2MoRALinear) and hasattr(module, 'router'):
                # The aux loss is computed during forward pass and returned
                # We need to store it during forward and retrieve it here
                if hasattr(module, '_last_aux_loss_a'):
                    total_aux_loss = total_aux_loss + module._last_aux_loss_a
                if hasattr(module, '_last_aux_loss_b'):
                    total_aux_loss = total_aux_loss + module._last_aux_loss_b
        
        return total_aux_loss * self.peft_config.load_balancing_loss_weight
    
    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, GD2MoRALayer):
                module.disable_adapters = False if enabled else True
    
    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)
    
    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


def mark_only_gd2mora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Mark only GD2MoRA parameters as trainable"""
    for n, p in model.named_parameters():
        if ("expert" not in n and "router" not in n and "lora_" not in n):
            p.requires_grad = False
    
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, GD2MoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Bias mode '{bias}' not implemented")


def compute_load_balancing_loss(router_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute load balancing loss to encourage uniform expert utilization.
    
    Args:
        router_probs: Router probability distribution over experts
        num_experts: Total number of experts
        
    Returns:
        Load balancing loss scalar
    """
    # Coefficient of variation squared
    return num_experts * torch.var(router_probs)
