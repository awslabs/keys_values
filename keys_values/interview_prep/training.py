from dataclasses import dataclass
import math
from typing import Literal, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from litgpt.utils import CycleIterator

from keys_values.data import Helmet, MyDataLoader, INPUT_IDS_NAME
from keys_values.finetune.batch_transform import (
    BatchTransformFactory,
    BatchTransform,
)
from keys_values.interview_prep.transformer import (
    Config,
    Transformer,
)


def create_state(
    config: Config,
) -> Dict[str, Any]:
    state = {
        "model": Transformer(config),
        "loss_function": torch.nn.CrossEntropyLoss(),  # TODO!
    }
    # TODO!
    return state


def fit(
    state: Dict[str, Any],
    train_dataloader: MyDataLoader,
    batch_transform: BatchTransform,
    max_num_epochs: int,
    max_num_steps: int,
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
    while num_steps < max_num_steps:
        batch = batch_transform(next(train_iterator))
        if train_iterator.epoch >= max_num_epochs:
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
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        print(f"Iteration {num_steps} (epoch {train_iterator.epoch}): loss = {loss.item()}")
        num_steps += 1
