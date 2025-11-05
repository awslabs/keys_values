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
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

class DistributedModelSync:
    """
    Efficient distributed model synchronization without extra memory allocation.
    Concatenates all model weights into a single tensor for efficient communication.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.param_shapes = []
        self.param_names = []
        self.total_params = 0
        
        # Pre-compute parameter metadata to avoid repeated calculations
        self._compute_param_metadata()
    
    def _compute_param_metadata(self):
        """Pre-compute parameter shapes and names for efficient reconstruction."""
        self.param_shapes = []
        self.param_names = []
        self.total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_shapes.append(param.shape)
                self.param_names.append(name)
                self.total_params += param.numel()
    
    def concatenate_weights(self) -> torch.Tensor:
        """
        Concatenate all model weights into a single flat tensor.
        Returns a view when possible to avoid memory allocation.
        """
        # Collect all parameter tensors
        param_tensors = []
        for param in self.model.parameters():
            if param.requires_grad:
                param_tensors.append(param.data.view(-1))
        
        # Concatenate into single tensor
        if len(param_tensors) == 1:
            return param_tensors[0]
        else:
            return torch.cat(param_tensors, dim=0)
    
    def reconstruct_weights(self, flat_tensor: torch.Tensor) -> None:
        """
        Reconstruct model weights from flat tensor in-place.
        No extra memory allocation - directly updates model parameters.
        """
        offset = 0
        param_dict = dict(self.model.named_parameters())
        
        for name, shape in zip(self.param_names, self.param_shapes):
            param = param_dict[name]
            param_size = param.numel()
            
            # Extract slice and reshape in-place
            param_slice = flat_tensor[offset:offset + param_size]
            param.data.copy_(param_slice.view(shape))
            
            offset += param_size
    
    def broadcast_weights(self, src_rank: int = 0) -> None:
        """
        Broadcast model weights from source rank to all other ranks.
        Uses concatenated tensor for efficient communication.
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        # Get current rank
        rank = dist.get_rank()
        
        if rank == src_rank:
            # Source rank: concatenate and broadcast
            flat_weights = self.concatenate_weights()
            dist.broadcast(flat_weights, src=src_rank)
        else:
            # Other ranks: create buffer and receive
            flat_weights = torch.zeros(self.total_params, 
                                     dtype=next(self.model.parameters()).dtype,
                                     device=self.device)
            dist.broadcast(flat_weights, src=src_rank)
            
            # Reconstruct model from received weights
            self.reconstruct_weights(flat_weights)
    
    def all_reduce_weights(self, op=dist.ReduceOp.AVG) -> None:
        """
        All-reduce model weights across all ranks (e.g., for gradient averaging).
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized")
        
        # Concatenate weights
        flat_weights = self.concatenate_weights()
        
        # All-reduce operation
        dist.all_reduce(flat_weights, op=op)
        
        # Reconstruct model from reduced weights
        self.reconstruct_weights(flat_weights)


class DistributedTrainer:
    """
    Complete distributed training wrapper with efficient weight synchronization.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.sync_handler = DistributedModelSync(model, device)
        
    def sync_model_from_rank(self, src_rank: int = 0):
        """Synchronize model weights from a specific rank."""
        self.sync_handler.broadcast_weights(src_rank)
    
    def average_gradients(self):
        """Average gradients across all processes."""
        if not dist.is_initialized():
            return
        
        # Collect gradients into flat tensor
        grad_tensors = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_tensors.append(param.grad.data.view(-1))
        
        if not grad_tensors:
            return
        
        # Concatenate and all-reduce gradients
        flat_grads = torch.cat(grad_tensors, dim=0)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG)
        
        # Reconstruct gradients in-place
        offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_size = param.grad.numel()
                param.grad.data.copy_(
                    flat_grads[offset:offset + param_size].view(param.grad.shape)
                )
                offset += param_size


def initialize_distributed_training(backend: str = 'nccl', 
                                  init_method: str = 'env://') -> Tuple[int, int]:
    """
    Initialize distributed training environment.
    
    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method
    
    Returns:
        Tuple of (rank, world_size)
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")
    
    dist.init_process_group(backend=backend, init_method=init_method)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Example usage and demonstration
def example_distributed_training():
    """
    Example of how to use the distributed training components.
    """
    # Initialize distributed environment
    rank, world_size = initialize_distributed_training()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(model, device)
    
    # Synchronize initial model weights from rank 0
    trainer.sync_model_from_rank(src_rank=0)
    
    # Example training step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Dummy data for demonstration
    inputs = torch.randn(32, 100, device=device)
    targets = torch.randint(0, 10, (32,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Average gradients across all processes
    trainer.average_gradients()
    
    # Update parameters
    optimizer.step()
    
    print(f"Rank {rank}: Loss = {loss.item():.4f}")
    
    # Cleanup
    cleanup_distributed()


# Advanced utility functions
def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, int]:
    """Get detailed memory footprint of model parameters."""
    memory_info = {
        'total_params': 0,
        'total_bytes': 0,
        'trainable_params': 0,
        'trainable_bytes': 0
    }
    
    for param in model.parameters():
        param_count = param.numel()
        param_bytes = param_count * param.element_size()
        
        memory_info['total_params'] += param_count
        memory_info['total_bytes'] += param_bytes
        
        if param.requires_grad:
            memory_info['trainable_params'] += param_count
            memory_info['trainable_bytes'] += param_bytes
    
    return memory_info


def verify_model_sync(models: List[torch.nn.Module]) -> bool:
    """
    Verify that all models have identical weights.
    Useful for debugging distributed synchronization.
    """
    if len(models) < 2:
        return True
    
    reference_model = models[0]
    ref_params = list(reference_model.parameters())
    
    for model in models[1:]:
        model_params = list(model.parameters())
        
        if len(ref_params) != len(model_params):
            return False
        
        for ref_param, model_param in zip(ref_params, model_params):
            if not torch.allclose(ref_param.data, model_param.data, rtol=1e-6):
                return False
    
    return True


# Memory-efficient checkpoint saving for distributed training
class DistributedCheckpoint:
    """Handle checkpointing in distributed training efficiently."""
    
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       filepath: str,
                       rank: int = 0) -> None:
        """Save checkpoint only from specified rank to avoid conflicts."""
        if dist.get_rank() == rank:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'rank': rank,
                'world_size': dist.get_world_size()
            }
            torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str, 
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """Load checkpoint and return metadata."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'rank': checkpoint.get('rank', 0),
            'world_size': checkpoint.get('world_size', 1)
        }


if __name__ == "__main__":
    # Run example (this would typically be launched with torchrun or similar)
    try:
        example_distributed_training()
    except RuntimeError as e:
        print(f"Note: {e}")
        print("To run distributed training, use: torchrun --nproc_per_node=N script.py")