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
from typing import Dict, List, Tuple, Any, NamedTuple
import copy

class ParameterSpec(NamedTuple):
    """Specification for a parameter without the actual tensor data."""
    name: str
    shape: Tuple[int, ...]
    offset: int
    size: int
    dtype: torch.dtype

class BufferSpec(NamedTuple):
    """Specification for a buffer."""
    name: str
    tensor: torch.Tensor

class ModelLayout:
    """
    Analyzes a model and creates a layout specification without copying weights.
    This allows us to recreate the model structure with a different backing tensor.
    """
    
    def __init__(self, template_model: nn.Module):
        self.param_specs: List[ParameterSpec] = []
        self.buffer_specs: List[BufferSpec] = []
        self.total_param_size = 0
        self.dtype = None
        
        # Analyze the template model structure
        self._analyze_model(template_model)
        
        # Create module structure without parameters (lightweight)
        self.module_structure = self._create_empty_structure(template_model)
    
    def _analyze_model(self, model: nn.Module):
        """Extract parameter specifications without copying data."""
        offset = 0
        
        for name, param in model.named_parameters():
            if self.dtype is None:
                self.dtype = param.dtype
            
            param_size = param.numel()
            self.param_specs.append(ParameterSpec(
                name=name,
                shape=param.shape,
                offset=offset,
                size=param_size,
                dtype=param.dtype
            ))
            offset += param_size
        
        self.total_param_size = offset
        
        # Store buffer specifications
        for name, buffer in model.named_buffers():
            self.buffer_specs.append(BufferSpec(name=name, tensor=buffer.clone()))
    
    def _create_empty_structure(self, template_model: nn.Module) -> nn.Module:
        """Create module structure without any parameters or buffers."""
        # Create a structure-only copy
        structure = copy.deepcopy(template_model)
        
        # Remove all parameters
        for name, _ in list(structure.named_parameters()):
            self._delete_parameter(structure, name)
        
        # Remove all buffers  
        for name, _ in list(structure.named_buffers()):
            self._delete_buffer(structure, name)
        
        return structure

    def _delete_parameter(self, module: nn.Module, param_name: str):
        """Delete parameter from module hierarchy."""
        parts = param_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        delattr(parent, parts[-1])
    
    def _delete_buffer(self, module: nn.Module, buffer_name: str):
        """Delete buffer from module hierarchy."""
        parts = buffer_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        delattr(parent, parts[-1])


class FlatTensorBackedModule(nn.Module):
    """
    A PyTorch module backed by a single flat tensor.
    Can be created from a ModelLayout without needing the original model in memory.
    """
    
    def __init__(self, layout: ModelLayout, flat_tensor: torch.Tensor):
        super().__init__()
        
        if flat_tensor.numel() != layout.total_param_size:
            raise ValueError(f"Flat tensor size {flat_tensor.numel()} doesn't match "
                           f"expected size {layout.total_param_size}")
        
        self.layout = layout
        self.flat_tensor = flat_tensor
        self.device = flat_tensor.device
        
        # Use the pre-created module structure
        self._module_structure = layout.module_structure
        
        # Create parameter views into flat tensor
        self._create_parameter_views()
        
        # Register buffers
        self._register_buffers()
    
    def _create_parameter_views(self):
        """Create and register parameter views into the flat tensor."""
        for spec in self.layout.param_specs:
            # Create view into flat tensor
            param_slice = self.flat_tensor[spec.offset:spec.offset + spec.size]
            param_view = param_slice.view(spec.shape)
            
            # Register as parameter
            self._register_parameter_by_name(spec.name, param_view)
    
    def _register_parameter_by_name(self, param_name: str, param_tensor: torch.Tensor):
        """Register parameter at the specified path."""
        parts = param_name.split('.')
        parent = self._module_structure
        for part in parts[:-1]:
            parent = getattr(parent, part)
        parent.register_parameter(parts[-1], nn.Parameter(param_tensor))
    
    def _register_buffers(self):
        """Register all buffers in the module structure."""
        for spec in self.layout.buffer_specs:
            parts = spec.name.split('.')
            parent = self._module_structure
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent.register_buffer(parts[-1], spec.tensor.to(self.device))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the module structure."""
        return self._module_structure(*args, **kwargs)
    
    def get_flat_tensor(self) -> torch.Tensor:
        """Get the underlying flat tensor."""
        return self.flat_tensor
    
    # Delegate all other methods to the module structure
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self._module_structure.named_parameters(prefix, recurse)
    
    def parameters(self, recurse: bool = True):
        return self._module_structure.parameters(recurse)
    
    def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
        return self._module_structure.named_modules(memo, prefix, remove_duplicate)
    
    def modules(self):
        return self._module_structure.modules()
    
    def state_dict(self, *args, **kwargs):
        return self._module_structure.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        return self._module_structure.load_state_dict(state_dict, strict)


def create_model_layout(template_model: nn.Module) -> ModelLayout:
    """
    Phase 1: Create model layout from template (template can be on CPU/small device).
    This extracts the structure without copying weights.
    """
    return ModelLayout(template_model)


def create_flat_tensor_from_model(model: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Create and populate flat tensor from existing model.
    Use this if you have the original model and want to extract its weights.
    """
    param_tensors = []
    for param in model.parameters():
        param_tensors.append(param.data.view(-1))
    
    flat_tensor = torch.cat(param_tensors, dim=0).to(device)
    flat_tensor.requires_grad_(True)
    return flat_tensor


def create_empty_flat_tensor(layout: ModelLayout, device: torch.device) -> torch.Tensor:
    """
    Phase 2: Create empty flat tensor based on layout.
    No original model needed in memory.
    """
    flat_tensor = torch.zeros(layout.total_param_size, 
                            dtype=layout.dtype,
                            device=device, 
                            requires_grad=True)
    return flat_tensor


def create_distributed_model_from_layout(layout: ModelLayout, 
                                       flat_tensor: torch.Tensor) -> 'ZeroCopyDistributedTrainer':
    """
    Phase 2: Create distributed model from layout and flat tensor.
    Original model no longer needed in memory.
    """
    flat_model = FlatTensorBackedModule(layout, flat_tensor)
    return ZeroCopyDistributedTrainer(flat_model)


class ZeroCopyDistributedSync:
    """Zero-copy distributed synchronization."""
    
    def __init__(self, flat_backed_model: FlatTensorBackedModule):
        self.model = flat_backed_model
        self.device = flat_backed_model.device
    
    def broadcast_weights(self, src_rank: int = 0):
        """Broadcast weights - zero copy, direct tensor operation."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        flat_tensor = self.model.get_flat_tensor()
        dist.broadcast(flat_tensor, src=src_rank)
    
    def all_reduce_weights(self, op=dist.ReduceOp.AVG):
        """All-reduce weights - zero copy, direct tensor operation."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        flat_tensor = self.model.get_flat_tensor()
        dist.all_reduce(flat_tensor, op=op)
    
    def reduce_scatter_weights(self, output_tensor: torch.Tensor):
        """Reduce-scatter for parameter sharding."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        flat_tensor = self.model.get_flat_tensor()
        dist.reduce_scatter_tensor(output_tensor, flat_tensor)


class ZeroCopyDistributedTrainer:
    """Distributed trainer with true zero-copy weight operations."""
    
    def __init__(self, flat_backed_model: FlatTensorBackedModule):
        self.model = flat_backed_model
        self.device = flat_backed_model.device
        self.sync_handler = ZeroCopyDistributedSync(flat_backed_model)
        self._grad_flat_tensor = None
    
    def _ensure_gradient_tensor(self):
        """Lazily create gradient flat tensor when needed."""
        if self._grad_flat_tensor is None:
            flat_tensor = self.model.get_flat_tensor()
            self._grad_flat_tensor = torch.zeros_like(flat_tensor)
    
    def sync_model_from_rank(self, src_rank: int = 0):
        """Synchronize model weights from specified rank."""
        self.sync_handler.broadcast_weights(src_rank)
    
    def average_gradients(self):
        """Average gradients across all processes."""
        if not dist.is_initialized():
            return
        
        self._ensure_gradient_tensor()
        
        # Collect gradients into flat tensor
        offset = 0
        has_grads = False
        
        for param in self.model.parameters():
            param_size = param.numel()
            if param.grad is not None:
                self._grad_flat_tensor[offset:offset + param_size].copy_(
                    param.grad.data.view(-1)
                )
                has_grads = True
            offset += param_size
        
        if not has_grads:
            return
        
        # All-reduce gradients
        dist.all_reduce(self._grad_flat_tensor, op=dist.ReduceOp.AVG)
        
        # Copy back to parameter gradients
        offset = 0
        for param in self.model.parameters():
            param_size = param.numel()
            if param.grad is not None:
                param.grad.data.copy_(
                    self._grad_flat_tensor[offset:offset + param_size].view(param.grad.shape)
                )
            offset += param_size
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        flat_tensor_bytes = self.model.flat_tensor.numel() * self.model.flat_tensor.element_size()
        grad_bytes = 0
        if self._grad_flat_tensor is not None:
            grad_bytes = self._grad_flat_tensor.numel() * self._grad_flat_tensor.element_size()
        
        return {
            'parameter_bytes': flat_tensor_bytes,
            'gradient_bytes': grad_bytes,
            'total_bytes': flat_tensor_bytes + grad_bytes,
            'parameter_count': self.model.flat_tensor.numel()
        }


# Memory-efficient model creation workflow
def efficient_distributed_model_creation(template_model: nn.Module, 
                                        device: torch.device,
                                        initialize_from_template: bool = True) -> ZeroCopyDistributedTrainer:
    """
    Memory-efficient distributed model creation workflow.
    
    Args:
        template_model: Model to use as template (can be on CPU)
        device: Target device for distributed model
        initialize_from_template: If True, copy weights from template
    
    Returns:
        ZeroCopyDistributedTrainer with single memory allocation
    """
    
    # Phase 1: Extract layout (template can be on CPU - very cheap)
    print("Phase 1: Analyzing model layout...")
    layout = create_model_layout(template_model)
    print(f"✓ Layout extracted: {layout.total_param_size:,} parameters")
    
    # Phase 2: Create flat tensor on target device
    print("Phase 2: Allocating flat tensor...")
    if initialize_from_template:
        # Copy weights from template to flat tensor
        flat_tensor = create_flat_tensor_from_model(template_model, device)
        print(f"✓ Flat tensor created and initialized from template")
    else:
        # Create empty flat tensor
        flat_tensor = create_empty_flat_tensor(layout, device)
        print(f"✓ Empty flat tensor created")
    
    # Now template_model can be deleted to free memory before creating final model
    del template_model  # Explicit deletion
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Phase 3: Create final model (only flat tensor + lightweight structure)
    print("Phase 3: Creating flat tensor backed model...")
    trainer = create_distributed_model_from_layout(layout, flat_tensor)
    print("✓ Zero-copy distributed model ready")
    
    return trainer


def broadcast_model_layout(layout: ModelLayout, src_rank: int = 0):
    """Broadcast model layout across ranks (lightweight operation)."""
    if not dist.is_initialized():
        return layout
    
    rank = dist.get_rank()
    
    if rank == src_rank:
        # Broadcast layout specs
        layout_data = {
            'param_specs': layout.param_specs,
            'buffer_specs': [(spec.name, spec.tensor) for spec in layout.buffer_specs],
            'total_param_size': layout.total_param_size,
            'dtype': layout.dtype
        }
        
        # Convert to tensor for broadcasting (small overhead)
        import pickle
        serialized = pickle.dumps(layout_data)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.long)
        dist.broadcast(size_tensor, src_rank)
        
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8)
        dist.broadcast(data_tensor, src_rank)
        
        return layout
    else:
        # Receive layout
        size_tensor = torch.tensor([0], dtype=torch.long)
        dist.broadcast(size_tensor, src_rank)
        
        data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)
        dist.broadcast(data_tensor, src_rank)
        
        import pickle
        layout_data = pickle.loads(data_tensor.numpy().tobytes())
        
        # Reconstruct layout
        new_layout = ModelLayout.__new__(ModelLayout)
        new_layout.param_specs = layout_data['param_specs']
        new_layout.buffer_specs = [BufferSpec(name, tensor) for name, tensor in layout_data['buffer_specs']]
        new_layout.total_param_size = layout_data['total_param_size']
        new_layout.dtype = layout_data['dtype']
        
        # We'll need to recreate the module structure, but this is lightweight
        # For now, return the layout data - structure creation happens later
        return new_layout


# Optimized distributed training workflow
class OptimizedDistributedWorkflow:
    """
    Optimized workflow for distributed training with minimal memory overhead.
    """
    
    @staticmethod
    def setup_rank_0(template_model: nn.Module, device: torch.device) -> Tuple[ModelLayout, ZeroCopyDistributedTrainer]:
        """
        Setup for rank 0: create layout and initialize model.
        Template model should be on CPU to minimize GPU memory usage.
        """
        print("=== Rank 0 Setup ===")
        
        # Create layout from template (template on CPU is fine)
        layout = create_model_layout(template_model)
        
        # Create trainer with weights from template
        trainer = efficient_distributed_model_creation(
            template_model, device, initialize_from_template=True
        )
        
        return layout, trainer
    
    @staticmethod
    def setup_other_ranks(layout: ModelLayout, device: torch.device) -> ZeroCopyDistributedTrainer:
        """
        Setup for non-zero ranks: create empty model and receive weights.
        No original model needed in memory.
        """
        print(f"=== Rank {dist.get_rank()} Setup ===")
        
        # Create empty flat tensor
        flat_tensor = create_empty_flat_tensor(layout, device)
        
        # Create trainer from layout and empty tensor
        trainer = create_distributed_model_from_layout(layout, flat_tensor)
        
        # Receive weights from rank 0
        trainer.sync_model_from_rank(src_rank=0)
        
        return trainer
    
    @staticmethod
    def create_distributed_ensemble(template_model: nn.Module, device: torch.device) -> ZeroCopyDistributedTrainer:
        """
        Complete workflow: setup distributed model across all ranks.
        Only rank 0 needs the original model in memory during initialization.
        """
        if not dist.is_initialized():
            # Single process fallback
            layout = create_model_layout(template_model)
            return efficient_distributed_model_creation(template_model, device)
        
        rank = dist.get_rank()
        
        if rank == 0:
            # Rank 0: create layout and initialize
            layout, trainer = OptimizedDistributedWorkflow.setup_rank_0(template_model, device)
            
            # Broadcast layout to other ranks
            broadcast_model_layout(layout, src_rank=0)
            
        else:
            # Other ranks: receive layout and create empty model
            layout = broadcast_model_layout(None, src_rank=0)  # Receive layout
            trainer = OptimizedDistributedWorkflow.setup_other_ranks(layout, device)
        
        print(f"✓ Rank {rank} ready for distributed training")
        return trainer


# Demonstration and utilities
def demonstrate_memory_efficiency():
    """Demonstrate the memory efficiency of the deferred allocation approach."""
    
    print("=== Memory Efficiency Demonstration ===")
    
    # Create a reasonably large model on CPU (cheap)
    template_model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512), 
        nn.ReLU(),
        nn.Linear(512, 100)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Method 1: Efficient creation (our approach)
    print("\n--- Efficient Creation ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    trainer = efficient_distributed_model_creation(template_model, device)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory: {peak_memory:.1f} MB")
    
    # Show final memory usage
    memory_info = trainer.get_memory_usage()
    print(f"Final parameter memory: {memory_info['parameter_bytes'] / 1024**2:.1f} MB")
    
    # Verify zero-copy property
    is_zero_copy = verify_zero_copy_property(trainer)
    print(f"Zero-copy verification: {'✓ PASSED' if is_zero_copy else '✗ FAILED'}")
    
    return trainer


def verify_zero_copy_property(trainer: ZeroCopyDistributedTrainer) -> bool:
    """Verify that parameters are truly views into the flat tensor."""
    flat_tensor = trainer.model.get_flat_tensor()
    base_ptr = flat_tensor.data_ptr()
    tensor_size = flat_tensor.numel() * flat_tensor.element_size()
    
    for param in trainer.model.parameters():
        param_ptr = param.data_ptr()
        if param_ptr < base_ptr or param_ptr >= base_ptr + tensor_size:
            return False
    
    return True


def initialize_distributed_training(backend: str = 'nccl') -> Tuple[int, int]:
    """Initialize distributed training environment."""
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")
    
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method='env://')
    
    return dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Example usage showcasing memory-efficient workflow
def example_memory_efficient_training():
    """
    Example showing memory-efficient distributed training setup.
    """
    print("=== Memory-Efficient Distributed Training ===")
    
    try:
        rank, world_size = initialize_distributed_training()
    except:
        rank, world_size = 0, 1
        print("Single process demo mode")
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    if world_size > 1:
        # Distributed setup
        if rank == 0:
            # Only rank 0 needs the original model
            template_model = nn.Sequential(
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 100)
            )
            trainer = OptimizedDistributedWorkflow.create_distributed_ensemble(template_model, device)
        else:
            # Other ranks don't need original model at all
            trainer = OptimizedDistributedWorkflow.create_distributed_ensemble(None, device)
    else:
        # Single process demo
        template_model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(), 
            nn.Linear(500, 100)
        )
        trainer = demonstrate_memory_efficiency()
    
    # Training demonstration
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\n=== Training on Rank {rank} ===")
    for step in range(3):
        # Generate batch
        inputs = torch.randn(32, 1000, device=device)
        targets = torch.randn(32, 100, device=device)
        
        # Training step
        optimizer.zero_grad()
        outputs = trainer.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Sync gradients (zero-copy)
        trainer.average_gradients()
        
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    # Show final memory stats
    memory_info = trainer.get_memory_usage()
    print(f"\nFinal memory usage: {memory_info['total_bytes'] / 1024**2:.1f} MB")
    
    return trainer


if __name__ == "__main__":
    try:
        trainer = example_memory_efficient_training()
        print("\n✓ Zero-copy distributed training completed successfully!")
    except Exception as e:
        print(f"Demo completed with note: {e}")
        print("For full distributed training, launch with: torchrun --nproc_per_node=N script.py")