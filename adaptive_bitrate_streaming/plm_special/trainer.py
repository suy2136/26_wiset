import numpy as np
import torch
import time
import csv
import copy
import json
import math
import os

from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch


def ensure_trainable_parameters_fp32(model):
    """Keep the frozen PLM dtype intact while promoting trainable weights.

    PEFT versions differ in whether newly-created adapter parameters inherit
    the base model dtype.  NBS training relies on FP32 adapter/head updates,
    so make that invariant explicit without touching frozen FP16 Llama
    weights.
    """
    promoted = []
    trainable = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            parameter.data = parameter.data.float()
            promoted.append({
                'parameter': name,
                'from_dtype': str(previous_dtype),
                'to_dtype': str(parameter.dtype),
            })
        trainable.append((name, parameter))
    invalid = [
        {'parameter': name, 'dtype': str(parameter.dtype)}
        for name, parameter in trainable
        if parameter.dtype != torch.float32
    ]
    if invalid:
        raise TypeError(
            f'NBS trainable parameters must be FP32: {invalid[:8]}'
        )
    return {
        'trainable_tensor_count': len(trainable),
        'trainable_parameter_count': sum(
            parameter.numel() for _, parameter in trainable
        ),
        'promoted_tensor_count': len(promoted),
        'promoted': promoted,
    }


class Trainer:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device,
                 batch_size=1, grad_accum_steps=1, lr_scheduler=None,
                 nbs_diagnostics_path=None, nbs_numeric_log_path=None,
                 rollback_lr_callback=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        self.rollback_lr_callback = rollback_lr_callback
        self.optimizer_step = 0
        self.nbs_allocator = getattr(model.plm, 'nash_rank_allocator', None)
        self.nbs_diagnostics_path = nbs_diagnostics_path
        self.nbs_numeric_log_path = nbs_numeric_log_path
        self.skipped_nonfinite_updates = 0
        self.consecutive_nonfinite = 0
        self.invalid_nonfinite_validations = 0
        self.max_consecutive_nonfinite = getattr(
            args, 'nbs_max_consecutive_nonfinite', 3
        )
        self.optimizer_state_dtype_verified = False
        self.rollback_count = 0
        self.consecutive_rollbacks = 0
        self.max_consecutive_rollbacks = getattr(
            args, 'nbs_max_consecutive_rollbacks', 3
        )
        self.rollback_backup_device = getattr(
            args, 'nbs_rollback_backup_device', 'cpu'
        )
        self.max_rollback_backup_mib = getattr(
            args, 'nbs_max_rollback_backup_mib', 2048.0
        )
        self.update_ratio_warning = getattr(
            args, 'nbs_update_ratio_warning', 0.01
        )
        self.max_update_ratio = getattr(
            args, 'nbs_max_update_ratio', 0.05
        )
        self.update_ratio_floor = getattr(
            args, 'nbs_update_ratio_floor', 0.01
        )
        self.max_update_rms = getattr(
            args, 'nbs_max_update_rms', 0.01
        )
        scaler_enabled = bool(
            self.nbs_allocator is not None
            and getattr(args, 'fp16', False)
            and str(device).startswith('cuda')
            and torch.cuda.is_available()
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)
        if self.nbs_allocator is not None:
            self._write_nbs_diagnostics(
                self.nbs_allocator.snapshot_diagnostics(
                    step=0, event='initialization'
                )
            )
            self._record_numeric_event(
                'training_start',
                grad_scaler_enabled=self.grad_scaler.is_enabled(),
                max_consecutive_nonfinite=self.max_consecutive_nonfinite,
                max_consecutive_rollbacks=self.max_consecutive_rollbacks,
                rollback_backup_device=self.rollback_backup_device,
                max_rollback_backup_mib=self.max_rollback_backup_mib,
                update_ratio_warning=self.update_ratio_warning,
                max_update_ratio=self.max_update_ratio,
                update_ratio_floor=self.update_ratio_floor,
                max_update_rms=self.max_update_rms,
            )

    def record_trainable_dtype_summary(self, summary):
        if self.nbs_allocator is None:
            return
        self._record_numeric_event('trainable_dtype_ready', **summary)

    def _transaction_parameters(self):
        return [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]

    def _snapshot_optimizer_transaction(self):
        """Clone only trainable tensors participating in this optimizer step."""
        parameters = self._transaction_parameters()
        parameter_set = set(parameters)
        parameter_names = {
            parameter: name for name, parameter in self.model.named_parameters()
            if parameter in parameter_set
        }
        estimated_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        )
        estimated_bytes += sum(
            value.numel() * value.element_size()
            for parameter in parameters
            for value in self.optimizer.state.get(parameter, {}).values()
            if torch.is_tensor(value)
        )
        estimated_mib = estimated_bytes / (1024 ** 2)
        if estimated_mib > self.max_rollback_backup_mib:
            raise MemoryError(
                'NBS rollback backup would require '
                f'{estimated_mib:.1f} MiB, above configured limit '
                f'{self.max_rollback_backup_mib:.1f} MiB'
            )
        backup_bytes = 0

        def backup_tensor(value):
            nonlocal backup_bytes
            backup_bytes += value.numel() * value.element_size()
            if self.rollback_backup_device == 'cpu':
                return value.detach().to(device='cpu', copy=True)
            return value.detach().clone()

        parameter_values = {
            parameter: backup_tensor(parameter) for parameter in parameters
        }
        optimizer_states = {}
        missing_states = set()
        for parameter in parameters:
            state = self.optimizer.state.get(parameter)
            if state is None:
                missing_states.add(parameter)
                continue
            optimizer_states[parameter] = {
                key: (
                    {
                        'tensor': backup_tensor(value),
                        'device': value.device,
                        'dtype': value.dtype,
                    }
                    if torch.is_tensor(value) else copy.deepcopy(value)
                )
                for key, value in state.items()
            }
        return {
            'parameters': parameter_values,
            'parameter_names': parameter_names,
            'optimizer_states': optimizer_states,
            'missing_states': missing_states,
            'backup_bytes': backup_bytes,
        }

    def _restore_optimizer_transaction(self, snapshot):
        with torch.no_grad():
            for parameter, previous in snapshot['parameters'].items():
                parameter.copy_(previous)
        for parameter in snapshot['missing_states']:
            self.optimizer.state.pop(parameter, None)
        for parameter, state in snapshot['optimizer_states'].items():
            current_state = self.optimizer.state.setdefault(parameter, {})
            for key in list(current_state):
                if key not in state:
                    del current_state[key]
            for key, value in state.items():
                if isinstance(value, dict) and 'tensor' in value:
                    current = current_state.get(key)
                    if (
                        torch.is_tensor(current)
                        and current.device == value['device']
                        and current.dtype == value['dtype']
                        and current.shape == value['tensor'].shape
                    ):
                        current.copy_(value['tensor'])
                    else:
                        current_state[key] = value['tensor'].to(
                            device=value['device'], dtype=value['dtype'],
                            copy=True,
                        )
                else:
                    current_state[key] = copy.deepcopy(value)

    def _update_ratio_diagnostics(self, snapshot):
        rows = []
        issues = []
        with torch.no_grad():
            for parameter, previous in snapshot['parameters'].items():
                previous_local = previous.to(
                    device=parameter.device, dtype=parameter.dtype
                )
                current = parameter.detach()
                delta = current.float() - previous_local.float()
                old_rms = float(
                    previous_local.float().square().mean().sqrt().item()
                )
                update_rms = float(delta.square().mean().sqrt().item())
                ratio = update_rms / max(
                    old_rms, self.update_ratio_floor
                )
                row = {
                    'parameter': snapshot['parameter_names'].get(
                        parameter, '<unnamed>'
                    ),
                    'old_rms': old_rms,
                    'update_rms': update_rms,
                    'update_ratio': ratio,
                }
                rows.append(row)
                if (
                    not math.isfinite(ratio)
                    or not math.isfinite(update_rms)
                    or ratio > self.max_update_ratio
                    or update_rms > self.max_update_rms
                ):
                    issues.append(row)
        rows.sort(key=lambda row: row['update_ratio'], reverse=True)
        return {
            'max_update_ratio': rows[0]['update_ratio'] if rows else 0.0,
            'max_update_rms': max(
                (row['update_rms'] for row in rows), default=0.0
            ),
            'warning': bool(
                rows and rows[0]['update_ratio'] > self.update_ratio_warning
            ),
            'top_updates': rows[:5],
            'issues': issues[:20],
        }

    def _register_optimizer_rollback(self, event, **details):
        self.rollback_count += 1
        self.consecutive_rollbacks += 1
        self.skipped_nonfinite_updates += 1
        lr_change = None
        if self.rollback_lr_callback is not None:
            lr_change = self.rollback_lr_callback(self.optimizer_step)
        self._record_numeric_event(
            event,
            rollback_count=self.rollback_count,
            consecutive_rollbacks=self.consecutive_rollbacks,
            learning_rate_change=lr_change,
            **details,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if self.consecutive_rollbacks >= self.max_consecutive_rollbacks:
            raise FloatingPointError(
                f'{event} repeated {self.consecutive_rollbacks} times; '
                f'see {self.nbs_numeric_log_path}'
            )

    def _optimizer_state_issues(self, check_dtype=False):
        issues = []
        tensor_states = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            state = self.optimizer.state.get(parameter, {})
            for state_name, value in state.items():
                if not torch.is_tensor(value) or not value.is_floating_point():
                    continue
                tensor_states.append((name, state_name, value))
                if check_dtype and value.dtype != torch.float32:
                    issues.append({
                        'parameter': name,
                        'state': state_name,
                        'reason': 'non_fp32',
                        'dtype': str(value.dtype),
                    })
        # Avoid one GPU synchronization per Adam tensor on the healthy path.
        # Scalar ``step`` states may live on CPU, so reduce health by device.
        states_by_device = {}
        for entry in tensor_states:
            states_by_device.setdefault(entry[2].device, []).append(entry)
        unhealthy_devices = set()
        for device, entries in states_by_device.items():
            health = torch.stack([
                torch.isfinite(value).all() for _, _, value in entries
            ])
            if not bool(health.all().item()):
                unhealthy_devices.add(device)
        for name, state_name, value in tensor_states:
            if value.device not in unhealthy_devices:
                continue
            if not bool(torch.isfinite(value).all().item()):
                issues.append({
                    'parameter': name,
                    'state': state_name,
                    'reason': 'nonfinite',
                    'dtype': str(value.dtype),
                })
        return issues

    def _record_numeric_event(self, event, **details):
        if not self.nbs_numeric_log_path:
            return
        os.makedirs(os.path.dirname(self.nbs_numeric_log_path), exist_ok=True)
        payload = {
            'event': event,
            'wall_time': time.time(),
            'optimizer_step': self.optimizer_step,
            'skipped_nonfinite_updates': self.skipped_nonfinite_updates,
            **details,
        }
        with open(self.nbs_numeric_log_path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, sort_keys=True) + '\n')

    def _register_nonfinite(self, event, **details):
        self.skipped_nonfinite_updates += 1
        self.consecutive_nonfinite += 1
        self._record_numeric_event(event, **details)
        self.optimizer.zero_grad(set_to_none=True)
        if self.consecutive_nonfinite >= self.max_consecutive_nonfinite:
            raise FloatingPointError(
                f'{event} repeated {self.consecutive_nonfinite} times; '
                f'see {self.nbs_numeric_log_path}'
            )

    def record_invalid_validation(self, **details):
        self.invalid_nonfinite_validations += 1
        self._record_numeric_event(
            'validation_nonfinite',
            invalid_nonfinite_validations=self.invalid_nonfinite_validations,
            **details,
        )

    def _adalora_delta_issues(self):
        monitored = []
        for module_name, module in self.model.plm.named_modules():
            if not hasattr(module, '_nbs_last_precast_finite'):
                continue
            dtype = module._nbs_output_dtype
            dtype_limit = torch.finfo(dtype).max
            healthy = (
                module._nbs_last_delta_finite
                & module._nbs_last_precast_finite
                & (module._nbs_last_precast_absmax <= dtype_limit)
            )
            monitored.append((module_name, module, healthy, dtype_limit))
        if not monitored:
            return []
        health = torch.stack([item[2] for item in monitored])
        if bool(health.all().item()):
            return []
        issues = []
        for module_name, module, healthy, dtype_limit in monitored:
            if bool(healthy.item()):
                continue
            issues.append({
                'module': getattr(module, '_nbs_module_name', module_name),
                'output_dtype': str(module._nbs_output_dtype),
                'dtype_limit': float(dtype_limit),
                'delta_absmax': float(module._nbs_last_delta_absmax.item()),
                'precast_absmax': float(module._nbs_last_precast_absmax.item()),
                'delta_finite': bool(module._nbs_last_delta_finite.item()),
                'precast_finite': bool(module._nbs_last_precast_finite.item()),
            })
        return issues

    def _gradient_issues(self):
        parameters = [
            (name, parameter) for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            return [{'parameter': '<all>', 'reason': 'no_gradients'}]
        health = torch.stack([
            torch.isfinite(parameter.grad.detach()).all()
            for _, parameter in parameters
        ])
        if bool(health.all().item()):
            return []
        issues = []
        for name, parameter in parameters:
            gradient = parameter.grad.detach()
            if bool(torch.isfinite(gradient).all().item()):
                continue
            finite_values = gradient[torch.isfinite(gradient)]
            issues.append({
                'parameter': name,
                'finite_elements': int(finite_values.numel()),
                'total_elements': int(gradient.numel()),
                'finite_absmax': (
                    float(finite_values.abs().max().item())
                    if finite_values.numel() else None
                ),
            })
        return issues

    def _parameter_issues(self):
        parameters = [
            (name, parameter.detach())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            return []
        health = torch.stack([
            torch.isfinite(parameter).all() for _, parameter in parameters
        ])
        if bool(health.all().item()):
            return []
        return [
            {'parameter': name}
            for name, parameter in parameters
            if not bool(torch.isfinite(parameter).all().item())
        ]

    def _allocator_numeric_issues(self):
        if self.nbs_allocator is None:
            return []
        issues = []
        for name, value in self.nbs_allocator.sensitivity.items():
            if not math.isfinite(float(value)):
                issues.append({
                    'source': 'sensitivity', 'layer': name,
                    'value': repr(value),
                })
        tensor_values = []
        for name, shadow in self.nbs_allocator.spectral_shadow.items():
            tensor_values.append(('spectral_shadow', name, shadow))
            tensor_values.append(
                ('spectral_energy', name, shadow.detach().float().abs().square())
            )
            module = self.nbs_allocator.layers[name]
            tensor_values.append((
                'lora_E', name,
                module.lora_E[self.nbs_allocator.adapter_name].detach(),
            ))
        if tensor_values:
            health = torch.stack([
                torch.isfinite(value).all() for _, _, value in tensor_values
            ])
            if not bool(health.all().item()):
                for source, name, value in tensor_values:
                    if not bool(torch.isfinite(value).all().item()):
                        issues.append({'source': source, 'layer': name})
        if not issues:
            weights = self.nbs_allocator._weights()
            weight_values = list(weights.values())
            if (
                not all(math.isfinite(value) and value >= 0 for value in weight_values)
                or not math.isclose(sum(weight_values), 1.0, rel_tol=1e-5)
            ):
                issues.append({
                    'source': 'allocation_weights',
                    'weight_sum': sum(weight_values),
                })
        return issues

    @staticmethod
    def _adapter_parameter(container, adapter_name):
        parameter = container[adapter_name]
        return parameter.weight if hasattr(parameter, 'weight') else parameter

    def _reset_reallocated_optimizer_moments(self, previous_masks):
        allocator = self.nbs_allocator
        changed_slots = 0
        reset_tensors = 0
        for name, module in allocator.layers.items():
            previous = previous_masks[name].to(allocator.masks[name].device)
            changed = previous.ne(allocator.masks[name]).reshape(-1)
            if not bool(changed.any().item()):
                continue
            changed_slots += int(changed.sum().item())
            adapter_name = allocator.adapter_name
            parameters = (
                ('A', self._adapter_parameter(module.lora_A, adapter_name)),
                ('B', self._adapter_parameter(module.lora_B, adapter_name)),
                ('E', self._adapter_parameter(module.lora_E, adapter_name)),
            )
            for kind, parameter in parameters:
                state = self.optimizer.state.get(parameter, {})
                for state_name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    state_tensor = state.get(state_name)
                    if state_tensor is None or state_tensor.shape != parameter.shape:
                        continue
                    with torch.no_grad():
                        if kind == 'A':
                            state_tensor[changed, ...] = 0
                        elif kind == 'B':
                            state_tensor[..., changed] = 0
                        else:
                            state_tensor.reshape(-1)[changed] = 0
                    reset_tensors += 1
        return changed_slots, reset_tensors

    def _write_nbs_diagnostics(self, rows):
        if not self.nbs_diagnostics_path or not rows:
            return
        rows = [
            {
                **row,
                # ``alpha`` is the normalized Nash bargaining weight. Keep
                # both names so historical analyses and paper tables agree.
                'bargaining_weight': row.get('alpha'),
            }
            for row in rows
        ]
        os.makedirs(os.path.dirname(self.nbs_diagnostics_path), exist_ok=True)
        exists = os.path.isfile(self.nbs_diagnostics_path)
        fieldnames = list(rows[0].keys())
        if exists:
            with open(
                self.nbs_diagnostics_path, newline='', encoding='utf-8'
            ) as existing_stream:
                fieldnames = csv.DictReader(existing_stream).fieldnames or fieldnames
        with open(self.nbs_diagnostics_path, 'a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, extrasaction='ignore'
            )
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
        accumulated_steps = 0

        self.model.train()
        for step, batch in enumerate(self.dataloader):
            train_loss = self.train_step(batch)
            delta_issues = self._adalora_delta_issues()
            loss_is_finite = bool(torch.isfinite(train_loss.detach()).item())
            if delta_issues or not loss_is_finite:
                self._register_nonfinite(
                    'forward_nonfinite', batch_step=step,
                    loss=float(train_loss.detach().item()),
                    adalora_delta_issues=delta_issues,
                )
                accumulated_steps = 0
                continue
            train_losses.append(train_loss.item())

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(train_loss).backward()
            else:
                train_loss.backward()
            accumulated_steps += 1
            should_update = (
                accumulated_steps >= self.grad_accum_steps
                or (step + 1 == dataset_size)
            )
            if self.nbs_allocator is None:
                # Preserve the historical ABR training behavior for plain LoRA.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if should_update:
                if self.nbs_allocator is not None:
                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.unscale_(self.optimizer)
                    gradient_issues = self._gradient_issues()
                    if gradient_issues:
                        old_scale = (
                            float(self.grad_scaler.get_scale())
                            if self.grad_scaler.is_enabled() else None
                        )
                        if self.grad_scaler.is_enabled():
                            self.grad_scaler.update()
                        self._register_nonfinite(
                            'gradient_nonfinite', batch_step=step,
                            grad_scaler_before=old_scale,
                            grad_scaler_after=(
                                float(self.grad_scaler.get_scale())
                                if self.grad_scaler.is_enabled() else None
                            ),
                            gradient_issues=gradient_issues,
                        )
                        accumulated_steps = 0
                        continue
                    # Match NBS v19: measure the raw accumulated A/B gradients,
                    # then clip once immediately before the optimizer update.
                    previous_sensitivity = dict(
                        self.nbs_allocator.sensitivity
                    )
                    previous_ema_step = self.nbs_allocator.ema_step
                    self.nbs_allocator.update_sensitivity()
                    allocator_issues = self._allocator_numeric_issues()
                    if allocator_issues:
                        self._record_numeric_event(
                            'allocator_nonfinite_before_update',
                            batch_step=step, allocator_issues=allocator_issues,
                        )
                        raise FloatingPointError(
                            'NBS allocator became non-finite before optimizer.step()'
                        )
                    try:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), .25,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        if 'non-finite' not in str(exc).lower():
                            raise
                        self.nbs_allocator.sensitivity.update(
                            previous_sensitivity
                        )
                        self.nbs_allocator.ema_step = previous_ema_step
                        old_scale = (
                            float(self.grad_scaler.get_scale())
                            if self.grad_scaler.is_enabled() else None
                        )
                        if self.grad_scaler.is_enabled():
                            self.grad_scaler.update(
                                new_scale=max(old_scale / 2.0, 1.0)
                            )
                        self._register_nonfinite(
                            'gradient_norm_nonfinite', batch_step=step,
                            grad_scaler_before=old_scale,
                            grad_scaler_after=(
                                float(self.grad_scaler.get_scale())
                                if self.grad_scaler.is_enabled() else None
                            ),
                            error=str(exc),
                        )
                        accumulated_steps = 0
                        continue
                else:
                    gradient_norm = None
                transaction_started = time.perf_counter()
                transaction = (
                    self._snapshot_optimizer_transaction()
                    if self.nbs_allocator is not None else None
                )
                transaction_backup_ms = (
                    (time.perf_counter() - transaction_started) * 1000.0
                    if transaction is not None else 0.0
                )
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                if self.nbs_allocator is not None:
                    parameter_issues = self._parameter_issues()
                    optimizer_state_issues = self._optimizer_state_issues(
                        check_dtype=True
                    )
                    update_diagnostics = self._update_ratio_diagnostics(
                        transaction
                    )
                    transaction_validation_ms = (
                        time.perf_counter() - transaction_started
                    ) * 1000.0
                    if (
                        parameter_issues
                        or optimizer_state_issues
                        or update_diagnostics['issues']
                    ):
                        self._restore_optimizer_transaction(transaction)
                        self.nbs_allocator.sensitivity.update(
                            previous_sensitivity
                        )
                        self.nbs_allocator.ema_step = previous_ema_step
                        self._register_optimizer_rollback(
                            'optimizer_transaction_rollback',
                            batch_step=step,
                            parameter_issues=parameter_issues,
                            optimizer_state_issues=optimizer_state_issues,
                            update_diagnostics=update_diagnostics,
                            gradient_norm_before_clip=float(
                                gradient_norm.item()
                            ),
                            gradient_norm_after_clip=min(
                                float(gradient_norm.item()), 0.25
                            ),
                            transaction_backup_ms=transaction_backup_ms,
                            transaction_validation_ms=(
                                transaction_validation_ms
                            ),
                            transaction_backup_mib=(
                                transaction['backup_bytes'] / (1024 ** 2)
                            ),
                        )
                        accumulated_steps = 0
                        continue
                    if not self.optimizer_state_dtype_verified:
                        self.optimizer_state_dtype_verified = True
                        self._record_numeric_event(
                            'optimizer_state_dtype_ready',
                            dtype='torch.float32',
                        )
                    self._record_numeric_event(
                        'optimizer_update_validated',
                        batch_step=step,
                        gradient_norm_before_clip=float(
                            gradient_norm.item()
                        ),
                        gradient_norm_after_clip=min(
                            float(gradient_norm.item()), 0.25
                        ),
                        update_diagnostics=update_diagnostics,
                        transaction_backup_ms=transaction_backup_ms,
                        transaction_validation_ms=transaction_validation_ms,
                        transaction_backup_mib=(
                            transaction['backup_bytes'] / (1024 ** 2)
                        ),
                    )
                # Commit the logical step only after parameter and optimizer
                # state validation. Allocation and scheduling must never see
                # a rejected optimizer update.
                self.optimizer_step += 1
                if self.nbs_allocator is not None:
                    should_allocate = self.nbs_allocator.should_allocate(
                        self.optimizer_step
                    )
                    if should_allocate:
                        previous_masks = {
                            name: mask.detach().clone()
                            for name, mask in self.nbs_allocator.masks.items()
                        }
                        self.nbs_allocator.allocate(self.optimizer_step)
                        changed_slots, reset_tensors = (
                            self._reset_reallocated_optimizer_moments(
                                previous_masks
                            )
                        )
                        self._write_nbs_diagnostics(
                            self.nbs_allocator.last_diagnostics
                        )
                        self._record_numeric_event(
                            'rank_allocation',
                            changed_slots=changed_slots,
                            optimizer_moment_tensors_reset=reset_tensors,
                            gradient_norm=float(gradient_norm.item()),
                        )
                    else:
                        self.nbs_allocator.enforce_masks()
                    allocator_issues = self._allocator_numeric_issues()
                    if allocator_issues:
                        self._record_numeric_event(
                            'allocator_nonfinite_after_update',
                            batch_step=step, allocator_issues=allocator_issues,
                        )
                        raise FloatingPointError(
                            'NBS allocator became non-finite after optimizer.step()'
                        )
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
                self.consecutive_nonfinite = 0
                self.consecutive_rollbacks = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:                
                mean_train_loss = np.mean(train_losses)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/skipped_nonfinite_updates'] = self.skipped_nonfinite_updates
        logs['training/optimizer_rollbacks'] = self.rollback_count
        logs['training/invalid_nonfinite_validations'] = (
            self.invalid_nonfinite_validations
        )

        return logs, train_losses

    def train_step(self, batch):
        states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)
        actions_pred = self.model(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        loss = self.loss_fn(actions_pred, labels)
        return loss
