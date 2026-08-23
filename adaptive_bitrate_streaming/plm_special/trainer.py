import numpy as np
import torch
import time
import csv
import os

from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch


class Trainer:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device,
                 batch_size=1, grad_accum_steps=1, lr_scheduler=None,
                 nbs_diagnostics_path=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        self.optimizer_step = 0
        self.nbs_allocator = getattr(model.plm, 'nash_rank_allocator', None)
        self.nbs_diagnostics_path = nbs_diagnostics_path
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)
        if self.nbs_allocator is not None:
            self._write_nbs_diagnostics(
                self.nbs_allocator.snapshot_diagnostics(
                    step=0, event='initialization'
                )
            )

    def _write_nbs_diagnostics(self, rows):
        if not self.nbs_diagnostics_path or not rows:
            return
        os.makedirs(os.path.dirname(self.nbs_diagnostics_path), exist_ok=True)
        exists = os.path.isfile(self.nbs_diagnostics_path)
        with open(self.nbs_diagnostics_path, 'a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def snapshot_nbs(self, event='snapshot'):
        if self.nbs_allocator is None:
            return
        self._write_nbs_diagnostics(
            self.nbs_allocator.snapshot_diagnostics(
                step=self.optimizer_step, event=event
            )
        )

    def train_epoch(self, report_loss_per_steps=100):
        train_losses = []
        logs = dict()

        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        for step, batch in enumerate(self.dataloader):
            train_loss = self.train_step(batch)
            train_losses.append(train_loss.item())

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            should_update = (
                ((step + 1) % self.grad_accum_steps == 0)
                or (step + 1 == dataset_size)
            )
            if self.nbs_allocator is None:
                # Preserve the historical ABR training behavior for plain LoRA.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if should_update:
                if self.nbs_allocator is not None:
                    # Match NBS v19: measure the raw accumulated A/B gradients,
                    # then clip once immediately before the optimizer update.
                    self.nbs_allocator.update_sensitivity()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
                self.optimizer.step()
                self.optimizer_step += 1
                if self.nbs_allocator is not None:
                    should_allocate = self.nbs_allocator.should_allocate(
                        self.optimizer_step
                    )
                    if should_allocate:
                        self.nbs_allocator.allocate(self.optimizer_step)
                        self._write_nbs_diagnostics(
                            self.nbs_allocator.last_diagnostics
                        )
                    else:
                        self.nbs_allocator.enforce_masks()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:                
                mean_train_loss = np.mean(train_losses)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs, train_losses

    def train_step(self, batch):
        states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)
        actions_pred = self.model(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        loss = self.loss_fn(actions_pred, labels)
        return loss
