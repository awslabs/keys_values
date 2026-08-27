from typing import Optional, Dict, Any

import torch

from litgpt.config import Config
from litgpt.utils import CycleIterator

from keys_values.data import Helmet, MyDataLoader, INPUT_IDS_NAME
from keys_values.finetune.batch_transform import (
    BatchTransformFactory,
    BatchTransform,
)
from keys_values.interview_prep.transformer import Transformer


# TODO: loss_function may not be correct for our data!
def create_state(
    config: Config,
    max_num_steps: int,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "model": Transformer(config),
        "loss_function": torch.nn.CrossEntropyLoss(),  # CHECK!
    }
    # Optimizer: Adam with improved weight decay regularization
    state["optimizer"] = torch.optim.AdamW(
        state["model"].named_parameters(), lr=0.0005,
    )
    # Cosine annealing scheduler, no warm-up
    state["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
        state["optimizer"], T_max=max_num_steps,
    )
    return state


def fit(
    state: Dict[str, Any],
    train_dataloader: MyDataLoader,
    batch_transform: BatchTransform,
    max_num_steps: Optional[int],
    max_num_epochs: Optional[int],
):
    # Parts of state
    model = state["model"]
    loss_function = state["loss_function"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    # Cycles over epochs several times
    train_iterator = CycleIterator(train_dataloader)
    model.train()  # Switch into training mode
    num_steps = 0

    # Training loop
    while max_num_steps is None or num_steps < max_num_steps:
        batch = batch_transform(next(train_iterator))
        if max_num_epochs is not None and train_iterator.epoch >= max_num_epochs:
            break
        # Context width of model must be long enough to process data batch.
        # Note: The context width is not fixed, but can grow. This affects
        # the position encoding only.
        seq_length = batch[INPUT_IDS_NAME].shape[-1]
        if model.context_width is None or model.context_width < seq_length:
            model.set_context_width(seq_length)
        # Compute gradients
        logits = model(input_ids=batch[INPUT_IDS_NAME])
        loss = loss_function(logits, batch["targets"])
        loss.backward()
        # Update step: Both optimizer and learning rate scheduler
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"Iteration {num_steps} (epoch {train_iterator.epoch}): loss = {loss.item()}")
        num_steps += 1


# TODO:
# - Make everything compatible with LitGPT, certain model (e.g, Qwen3-0.6B)
#   But code from scratch
# - Load checkpoint with tokenizer
# - Use my own data loaders
# - Write tests: Must give same results as my lib!
def main(
    config: Config,
    max_num_steps: Optional[int] = None,
    max_num_epochs: Optional[int] = None,
):
    if max_num_steps is None and max_num_epochs is None:
        raise ValueError("One of `max_num_steps` or `max_num_epochs` must be specified.")
    # Create training data iterator
    train_dataloader = TODO
    batch_transform = TODO
    # Create state: Model, loss function, optimizer, LR scheduler
    if max_num_epochs is None:
        lr_max_steps = max_num_steps
    else:
        lr_max_steps = len(train_dataloader) * max_num_epochs
        if max_num_steps is not None:
            lr_max_steps = min(max_num_steps, lr_max_steps)
    state = create_state(config, max_num_steps=lr_max_steps)
    # TODO: Load checkpoint
    # - May have to convert namings: Some differ from LitGPT code
    # Run training
    fit(
        state=state,
        train_dataloader=train_dataloader,
        batch_transform=batch_transform,
        max_num_steps=max_num_steps,
        max_num_epochs=max_num_epochs,
    )
