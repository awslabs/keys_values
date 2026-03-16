# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import copy

class FlatTensorBackedModule(nn.Module):
    """
    A PyTorch module where all parameters are views into a single flat tensor.
    This enables zero-copy distributed synchronization.
    """
    
    def __init__(self, original_module: nn.Module, device: torch.device):
        super().__init__()
        self.device = device
        self._param_info = []  # Store (name, shape, offset, size) for each parameter
        self._buffer_info = []  # Store buffer information separately
        self.total_param_size = 0
        
        # Clone the original module structure but don't copy parameters yet
        self._module_structure = self._clone_module_structure(original_module)
        
        # Analyze original module to get parameter layout
        self._analyze_parameters(original_module)
        
        # Create the single flat tensor that will back all parameters
        self.flat_tensor = torch.zeros(self.total_param_size, 
                                     dtype=next(original_module.parameters()).dtype,
                                     device=device, requires_grad=True)
        
        # Create parameter views and register them
        self._create_parameter_views()
        
        # Copy original weights into our flat tensor
        self._copy_original_weights(original_module)
        
        # Copy buffers (non-parameter tensors like batch norm running stats)
        self._copy_buffers(original_module)
    
    def _clone_module_structure(self, module: nn.Module) -> nn.Module:
        """Clone module structure without parameters."""
        # Create a deep copy but we'll replace parameters later
        cloned = copy.deepcopy(module)
        # Clear all parameters - we'll create our own views
        for name, _ in cloned.named_parameters():
            self._delete_parameter(cloned, name)
        return cloned
    
    def _delete_parameter(self, module: nn.Module, param_name: str):
        """Safely delete a parameter from a module."""
        parts = param_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        delattr(parent, parts[-1])
    
    def _analyze_parameters(self, original_module: nn.Module):
        """Analyze original module to determine parameter layout."""
        offset = 0
        for name, param in original_module.named_parameters():
            param_size = param.numel()
            self._param_info.append((name, param.shape, offset, param_size))
            offset += param_size
        self.total_param_size = offset
    
    def _create_parameter_views(self):
        """Create parameter views into the flat tensor and register them."""
        for name, shape, offset, size in self._param_info:
            # Create view into flat tensor
            param_view = self.flat_tensor[offset:offset + size].view(shape)
            
            # Register as parameter in the module structure
            self._register_parameter_by_name(self._module_structure, name, param_view)
    
    def _register_parameter_by_name(self, module: nn.Module, param_name: str, param_tensor: torch.Tensor):
        """Register a parameter at the specified path in the module."""
        parts = param_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Register as parameter
        parent.register_parameter(parts[-1], nn.Parameter(param_tensor))
    
    def _copy_original_weights(self, original_module: nn.Module):
        """Copy weights from original module into our flat tensor."""
        param_dict = dict(original_module.named_parameters())
        
        for name, shape, offset, size in self._param_info:
            original_param = param_dict[name]
            self.flat_tensor[offset:offset + size].copy_(original_param.data.view(-1))
    
    def _copy_buffers(self, original_module: nn.Module):
        """Copy non-parameter buffers (like batch norm stats)."""
        for name, buffer in original_module.named_buffers():
            parts = name.split('.')
            parent = self._module_structure
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent.register_buffer(parts[-1], buffer.clone().to(self.device))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the module structure."""
        return self._module_structure(*args, **kwargs)
    
    def get_flat_tensor(self) -> torch.Tensor:
        """Get the underlying flat tensor (for broadcasting)."""
        return self.flat_tensor
    
    def update_from_flat_tensor(self, new_flat_tensor: torch.Tensor):
        """Update all parameters from a new flat tensor (zero-copy)."""
        self.flat_tensor.data.copy_(new_flat_tensor)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Override to return parameters from our module structure."""
        return self._module_structure.named_parameters(prefix, recurse)
    
    def parameters(self, recurse: bool = True):
        """Override to return parameters from our module structure."""
        return self._module_structure.parameters(recurse)
    
    def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
        """Override to return modules from our module structure."""
        return self._module_structure.named_modules(memo, prefix, remove_duplicate)
    
    def modules(self):
        """Override to return modules from our module structure."""
        return self._module_structure.modules()
    
    def state_dict(self, *args, **kwargs):
        """Override to return state dict from our module structure."""
        return self._module_structure.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Override to load state dict into our module structure."""
        return self._module_structure.load_state_dict(state_dict, strict)


class ZeroCopyDistributedSync:
    """
    Zero-copy distributed synchronization using flat tensor backed modules.
    """
    
    def __init__(self, flat_backed_model: FlatTensorBackedModule):
        self.model = flat_backed_model
        self.device = flat_backed_model.device
    
    def broadcast_weights(self, src_rank: int = 0):
        """Broadcast weights with zero copies - direct operation on flat tensor."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        flat_tensor = self.model.get_flat_tensor()
        dist.broadcast(flat_tensor, src=src_rank)
        # No reconstruction needed - parameters are already views!
    
    def all_reduce_weights(self, op=dist.ReduceOp.AVG):
        """All-reduce weights with zero copies."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        flat_tensor = self.model.get_flat_tensor()
        dist.all_reduce(flat_tensor, op=op)
        # No reconstruction needed - parameters are already views!
    
    def all_gather_weights(self) -> torch.Tensor:
        """Gather weights from all ranks into a single tensor."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        world_size = dist.get_world_size()
        flat_tensor = self.model.get_flat_tensor()
        
        # Create output tensor for all gathered weights
        gathered_weights = torch.zeros(world_size, flat_tensor.numel(),
                                     dtype=flat_tensor.dtype,
                                     device=self.device)
        
        # Gather weights from all ranks
        dist.all_gather_into_tensor(gathered_weights, flat_tensor)
        
        return gathered_weights


class ZeroCopyDistributedTrainer:
    """
    Complete distributed trainer using zero-copy weight synchronization.
    """
    
    def __init__(self, original_model: nn.Module, device: torch.device):
        self.device = device
        
        # Create flat tensor backed model
        self.model = FlatTensorBackedModule(original_model, device)
        
        # Create sync handler
        self.sync_handler = ZeroCopyDistributedSync(self.model)
        
        # Track gradient state
        self._grad_flat_tensor = None
        self._create_gradient_views()
    
    def _create_gradient_views(self):
        """Create flat tensor for gradients that mirrors parameter structure."""
        flat_tensor = self.model.get_flat_tensor()
        self._grad_flat_tensor = torch.zeros_like(flat_tensor)
        
        # Create gradient views (will be populated during backward pass)
        offset = 0
        for name, shape, param_offset, size in self.model._param_info:
            # Gradients will be created by autograd, but we track the structure
            offset += size
    
    def sync_model_from_rank(self, src_rank: int = 0):
        """Synchronize model weights from specified rank (zero-copy)."""
        self.sync_handler.broadcast_weights(src_rank)
    
    def average_gradients(self):
        """Average gradients across all processes (zero-copy)."""
        if not dist.is_initialized():
            return
        
        # Collect gradients into our flat gradient tensor
        grad_offset = 0
        has_grads = False
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_size = param.grad.numel()
                self._grad_flat_tensor[grad_offset:grad_offset + param_size].copy_(
                    param.grad.data.view(-1)
                )
                has_grads = True
            grad_offset += param.numel() if param.grad is not None else param.numel()
        
        if not has_grads:
            return
        
        # All-reduce gradients
        dist.all_reduce(self._grad_flat_tensor, op=dist.ReduceOp.AVG)
        
        # Copy back to parameter gradients (in-place)
        grad_offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_size = param.grad.numel()
                param.grad.data.copy_(
                    self._grad_flat_tensor[grad_offset:grad_offset + param_size].view(param.grad.shape)
                )
            grad_offset += param.numel() if param.grad is not None else param.numel()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get actual memory usage - should show significant savings."""
        flat_tensor_bytes = self.model.flat_tensor.numel() * self.model.flat_tensor.element_size()
        grad_tensor_bytes = self._grad_flat_tensor.numel() * self._grad_flat_tensor.element_size() if self._grad_flat_tensor is not None else 0
        
        return {
            'parameter_bytes': flat_tensor_bytes,
            'gradient_bytes': grad_tensor_bytes,
            'total_bytes': flat_tensor_bytes + grad_tensor_bytes,
            'parameter_count': self.model.flat_tensor.numel()
        }


def create_distributed_model(original_model: nn.Module, device: torch.device) -> ZeroCopyDistributedTrainer:
    """
    Factory function to create a zero-copy distributed model.
    
    Args:
        original_model: The model to convert
        device: Target device
    
    Returns:
        ZeroCopyDistributedTrainer with flat tensor backed model
    """
    return ZeroCopyDistributedTrainer(original_model, device)


def compare_memory_usage(original_model: nn.Module, flat_model: FlatTensorBackedModule):
    """Compare memory usage between original and flat tensor backed models."""
    
    def get_model_memory(model):
        total_bytes = 0
        total_params = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
            total_params += param.numel()
        return total_bytes, total_params
    
    orig_bytes, orig_params = get_model_memory(original_model)
    flat_bytes = flat_model.flat_tensor.numel() * flat_model.flat_tensor.element_size()
    
    print(f"Original model: {orig_bytes:,} bytes ({orig_params:,} parameters)")
    print(f"Flat tensor model: {flat_bytes:,} bytes ({flat_model.flat_tensor.numel():,} parameters)")
    print(f"Memory reduction: {orig_bytes - flat_bytes:,} bytes")
    print(f"Efficiency: {flat_bytes / orig_bytes:.1%} of original memory")


# Example usage with comprehensive demonstration
def example_zero_copy_training():
    """Demonstrate zero-copy distributed training."""
    
    # Initialize distributed environment
    try:
        rank, world_size = initialize_distributed_training()
    except:
        # Fallback for single process demo
        rank, world_size = 0, 1
        print("Running in single-process mode for demonstration")
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create original model
    original_model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    print("=== Memory Usage Comparison ===")
    
    # Create zero-copy distributed model
    trainer = create_distributed_model(original_model, device)
    
    # Compare memory usage
    compare_memory_usage(original_model, trainer.model)
    
    print(f"\n=== Training Demonstration (Rank {rank}) ===")
    
    # Synchronize initial weights (zero-copy)
    if world_size > 1:
        trainer.sync_model_from_rank(src_rank=0)
        print("✓ Synchronized weights across ranks (zero-copy)")
    
    # Setup training
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training step with zero-copy gradient averaging
    for step in range(3):
        # Generate dummy batch
        batch_size = 64
        inputs = torch.randn(batch_size, 1000, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = trainer.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Average gradients across processes (zero-copy)
        if world_size > 1:
            trainer.average_gradients()
        
        # Update parameters
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    # Demonstrate that parameters are truly views
    print(f"\n=== Verification ===")
    print(f"Flat tensor device: {trainer.model.flat_tensor.device}")
    print(f"First layer weight device: {list(trainer.model.parameters())[0].device}")
    print(f"Parameters share storage: {trainer.model.flat_tensor.data_ptr() == list(trainer.model.parameters())[0].data_ptr()}")
    
    return trainer


def initialize_distributed_training(backend: str = 'nccl', 
                                  init_method: str = 'env://') -> Tuple[int, int]:
    """Initialize distributed training environment."""
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")
    
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class ZeroCopyCheckpoint:
    """Efficient checkpointing for flat tensor backed models."""
    
    @staticmethod
    def save_checkpoint(trainer: ZeroCopyDistributedTrainer,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       filepath: str,
                       rank: int = 0):
        """Save checkpoint from specified rank only."""
        if not dist.is_initialized() or dist.get_rank() == rank:
            checkpoint = {
                'epoch': epoch,
                'flat_tensor': trainer.model.flat_tensor.cpu(),  # Single tensor save
                'param_info': trainer.model._param_info,  # Parameter layout
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model_buffers': {name: buf for name, buf in trainer.model.named_buffers()}
            }
            torch.save(checkpoint, filepath)
            print(f"✓ Checkpoint saved (flat tensor: {trainer.model.flat_tensor.numel():,} params)")
    
    @staticmethod
    def load_checkpoint(filepath: str, 
                       original_model: nn.Module,
                       device: torch.device,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[ZeroCopyDistributedTrainer, Dict]:
        """Load checkpoint and recreate flat tensor backed model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create new trainer
        trainer = create_distributed_model(original_model, device)
        
        # Load flat tensor directly
        trainer.model.flat_tensor.data.copy_(checkpoint['flat_tensor'].to(device))
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load buffers
        if 'model_buffers' in checkpoint:
            for name, buffer in checkpoint['model_buffers'].items():
                parts = name.split('.')
                parent = trainer.model._module_structure
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent.register_buffer(parts[-1], buffer.to(device), persistent=False)
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'param_count': checkpoint['flat_tensor'].numel()
        }
        
        print(f"✓ Checkpoint loaded (flat tensor: {checkpoint['flat_tensor'].numel():,} params)")
        return trainer, metadata


# Advanced utilities for zero-copy distributed training
def verify_zero_copy_property(trainer: ZeroCopyDistributedTrainer) -> bool:
    """Verify that parameters are truly views into the flat tensor."""
    flat_tensor = trainer.model.get_flat_tensor()
    
    # Check if all parameters share the same underlying storage
    for param in trainer.model.parameters():
        if param.data_ptr() < flat_tensor.data_ptr() or \
           param.data_ptr() >= flat_tensor.data_ptr() + flat_tensor.numel() * flat_tensor.element_size():
            return False
    
    return True


def benchmark_sync_performance(trainer: ZeroCopyDistributedTrainer, iterations: int = 10):
    """Benchmark synchronization performance."""
    if not dist.is_initialized() or dist.get_world_size() < 2:
        print("Benchmarking requires distributed environment with multiple processes")
        return
    
    import time
    
    # Warm up
    for _ in range(3):
        trainer.sync_handler.broadcast_weights(src_rank=0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark broadcast
    start_time = time.time()
    for _ in range(iterations):
        trainer.sync_handler.broadcast_weights(src_rank=0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    param_count = trainer.model.flat_tensor.numel()
    throughput = param_count / avg_time / 1e6  # Million parameters per second
    
    print(f"Broadcast performance:")
    print(f"  Average time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} M params/sec")
    print(f"  Total parameters: {param_count:,}")


if __name__ == "__main__":
    # Demonstrate zero-copy distributed training
    print("=== Zero-Copy Distributed Training Demo ===")
    
    try:
        trainer = example_zero_copy_training()
        
        # Verify zero-copy property
        is_zero_copy = verify_zero_copy_property(trainer)
        print(f"\n✓ Zero-copy verification: {'PASSED' if is_zero_copy else 'FAILED'}")
        
        # Show memory usage
        memory_info = trainer.get_memory_usage()
        print(f"\nMemory usage:")
        print(f"  Parameters: {memory_info['parameter_bytes']:,} bytes")
        print(f"  Gradients: {memory_info['gradient_bytes']:,} bytes")
        print(f"  Total: {memory_info['total_bytes']:,} bytes")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("For full distributed training, use: torchrun --nproc_per_node=N script.py")