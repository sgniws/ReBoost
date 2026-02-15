"""Training loop for Boosted Late Fusion: Sustained Boosting + ACA."""
import os
import sys
import numpy as np
import datetime
from tqdm import tqdm
import torch
from copy import deepcopy
import json
from typing import Optional, List, Dict, Any, Callable
from torch.utils.data import DataLoader


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '==========' * 8 + '%s' % nowtime)
    print(str(info) + '\n')


def _collect_modality_grad_stats_boosted(net, n_experts=4, eps=1e-8):
    """Collect gradient norms from first encoder stage. Uses backbones (BoostedLateFusion)."""
    acts = [None] * n_experts
    handles = []

    def make_hook(idx):
        def hook(module, input, output):
            output.retain_grad()
            acts[idx] = output
        return hook

    if not hasattr(net, 'backbones'):
        return None
    for i in range(min(n_experts, len(net.backbones))):
        backbone = net.backbones[i]
        if not hasattr(backbone, 'encoder_stages') or len(backbone.encoder_stages) == 0:
            return None
        h = backbone.encoder_stages[0].register_forward_hook(make_hook(i))
        handles.append(h)

    return acts, handles


class AdaptiveClassifierAssignment:
    """Adaptive Classifier Assignment: add head to weakest modality when confidence gap > tau."""

    def __init__(self, sigma=1.0, tau=0.01, check_interval_epochs=10, max_heads=10):
        self.sigma = sigma
        self.tau = tau
        self.check_interval = check_interval_epochs
        self.max_heads = max_heads

    @torch.no_grad()
    def compute_confidence_scores(self, model, dataloader, device) -> List[float]:
        """Compute per-modality confidence: mean sigmoid probability on foreground voxels."""
        model.eval()
        n_modalities = model.n_experts
        score_sums = [0.0] * n_modalities
        count = 0

        for batch in dataloader:
            x = batch['img'].to(device)
            y = batch['label'].to(device)
            output = model(x)

            for m_idx in range(n_modalities):
                combined_logits = output['modality_all_logits'][m_idx]
                combined_probs = torch.sigmoid(combined_logits)
                foreground_mask = (y > 0.5).float()
                masked_probs = combined_probs * foreground_mask
                fg_count = foreground_mask.sum(dim=[2, 3, 4]).clamp(min=1)
                sample_confidence = (
                    masked_probs.sum(dim=[2, 3, 4]) / fg_count
                ).mean(dim=1)
                score_sums[m_idx] += sample_confidence.sum().item()

            count += x.shape[0]

        scores = [s / max(count, 1) for s in score_sums]
        model.train()
        return scores

    def should_check(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.check_interval == 0

    def assign(self, model, scores: List[float]) -> Optional[int]:
        """If max - sigma*min > tau, add head to weakest modality. Return modality index or None."""
        s_max = max(scores)
        s_min = min(scores)
        weak_idx = scores.index(s_min)

        if s_max - self.sigma * s_min > self.tau:
            current_heads = model.get_num_heads()
            if current_heads[weak_idx] < self.max_heads:
                success = model.add_head(weak_idx)
                if success:
                    return weak_idx
        return None


class StepRunner:
    """Step runner for Boosted model: model returns dict, loss_fn returns (loss, details)."""

    def __init__(self, net, loss_fn, stage='train', metrics_dict=None,
                 optimizer=None, device='cuda:0', monitor_modality_grad=False):
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.device = device
        self.monitor_modality_grad = monitor_modality_grad

    def step(self, batch):
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        grad_stats = None
        if self.stage == 'train' and self.monitor_modality_grad and hasattr(self.net, 'backbones'):
            n_experts = self.net.n_experts
            hook_result = _collect_modality_grad_stats_boosted(self.net, n_experts)
            if hook_result is not None:
                acts, handles = hook_result
                try:
                    model_output = self.net(features)
                    total_loss, loss_details = self.loss_fn(model_output, labels)
                    total_loss.backward()

                    eps = 1e-8
                    norms = []
                    for i in range(n_experts):
                        t = acts[i]
                        if t is not None and t.grad is not None:
                            norms.append(t.grad.norm().item())
                        else:
                            norms.append(0.0)
                    total_n = sum(norms) + eps
                    ratios = [n / total_n for n in norms]
                    balance = min(norms) / (max(norms) + eps) if max(norms) > 0 else 0.0
                    grad_stats = {
                        'grad_norm_0': norms[0], 'grad_norm_1': norms[1],
                        'grad_norm_2': norms[2], 'grad_norm_3': norms[3],
                        'grad_ratio_0': ratios[0], 'grad_ratio_1': ratios[1],
                        'grad_ratio_2': ratios[2], 'grad_ratio_3': ratios[3],
                        'grad_balance': balance,
                    }
                    for k, v in loss_details.items():
                        grad_stats[k] = v
                finally:
                    for h in handles:
                        h.remove()
                self.optimizer.step()
                self.optimizer.zero_grad()
                return total_loss.item(), len(labels), grad_stats

        model_output = self.net(features)
        total_loss, loss_details = self.loss_fn(model_output, labels)
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        out_extra = loss_details
        return total_loss.item(), len(labels), out_extra

    def step_infer(self, batch):
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        model_output = self.net(features)
        total_loss, _ = self.loss_fn(model_output, labels)
        preds = model_output['output']

        step_metrics = {
            self.stage + '_' + name: metric_fn(preds, labels).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return total_loss.item(), step_metrics, len(labels)

    def train_step(self, batch):
        self.net.train()
        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.net.eval()
        return self.step_infer(batch)

    def __call__(self, batch):
        if self.stage == 'train':
            return self.train_step(batch)
        return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner: StepRunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def _train_handler(self):
        cur_loss = 0
        cur_num_sample = 0
        cur_extra = {}
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            result = self.steprunner(batch)
            step_loss, batch_size = result[0], result[1]
            extra = result[2] if len(result) == 3 else None

            cur_loss = (cur_num_sample * cur_loss + batch_size * step_loss) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size

            if extra is not None:
                for k, v in extra.items():
                    if isinstance(v, (int, float)):
                        cur_extra[k] = (cur_num_sample - batch_size) * cur_extra.get(k, 0) + batch_size * v
                for k in list(cur_extra.keys()):
                    cur_extra[k] = cur_extra[k] / cur_num_sample if cur_num_sample > 0 else cur_extra[k]

            cur_log = {self.stage + '_loss': cur_loss}
            for k, v in cur_extra.items():
                cur_log[self.stage + '_' + k] = v
            loop.set_postfix(**{k: v for k, v in cur_log.items() if k == self.stage + '_loss'})
        return cur_log

    def _eval_handler(self):
        cur_loss = 0
        cur_num_sample = 0
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            step_loss, step_metrics, batch_size = self.steprunner(batch)
            cur_loss = (cur_num_sample * cur_loss + batch_size * step_loss) / (cur_num_sample + batch_size)
            if step == 0:
                cur_metrics = step_metrics.copy()
            else:
                for metric_name, metric_value in step_metrics.items():
                    cur_metrics[metric_name] = (
                        cur_num_sample * cur_metrics[metric_name] + batch_size * metric_value
                    ) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size
            cur_log = dict({self.stage + '_loss': cur_loss}, **cur_metrics)
            loop.set_postfix(**cur_log)
        return cur_log

    def __call__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        if self.stage == 'train':
            return self._train_handler()
        return self._eval_handler()


def _fmt_log_line_boosted(epoch, task_loss, train_log, did_val, val_log=None, aca_added=None, n_heads=None):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    epoch_str = 'Epoch {:4d}'.format(epoch)
    parts = [f'[{ts}] {epoch_str}']

    if train_log.get('train_grad_norm_0') is not None:
        g0 = train_log.get('train_grad_norm_0', 0) or 0
        g1 = train_log.get('train_grad_norm_1', 0) or 0
        g2 = train_log.get('train_grad_norm_2', 0) or 0
        g3 = train_log.get('train_grad_norm_3', 0) or 0
        r0 = (train_log.get('train_grad_ratio_0', 0) or 0) * 100
        r1 = (train_log.get('train_grad_ratio_1', 0) or 0) * 100
        r2 = (train_log.get('train_grad_ratio_2', 0) or 0) * 100
        r3 = (train_log.get('train_grad_ratio_3', 0) or 0) * 100
        parts.append(f'grad_encoder: {g0:.4e}, {g1:.4e}, {g2:.4e}, {g3:.4e}')
        parts.append(f'encoder_ratio: {r0:.1f}%, {r1:.1f}%, {r2:.1f}%, {r3:.1f}%')

    if n_heads is not None:
        parts.append(f'n_heads: {n_heads}')

    parts.append(f'task_loss: {task_loss:.4f}')

    if aca_added is not None:
        parts.append(f'ACA_added_modality: {aca_added}')

    if did_val and val_log is not None:
        val_loss = val_log.get('val_loss')
        val_dice = val_log.get('val_dice')
        wt = val_log.get('val_dice_wt')
        tc = val_log.get('val_dice_tc')
        et = val_log.get('val_dice_et')
        if isinstance(val_loss, (int, float)):
            parts.append(f'val_loss: {val_loss:.4f}')
        if isinstance(val_dice, (int, float)):
            parts.append(f'val_dice: {val_dice:.4f}')
        if all(isinstance(x, (int, float)) for x in (wt, tc, et)):
            parts.append(f'WT/TC/ET: {wt:.4f}/{tc:.4f}/{et:.4f}')
    return ' | '.join(parts)


def train_model(
    net,
    optimizer,
    loss_fn,
    metrics_dict,
    train_data,
    val_data,
    val_freq=1,
    num_epoch=100,
    results_dir='results/0/',
    device='cuda:0',
    apply_early_stopping=False,
    patience=5,
    monitor='val_loss',
    eval_mode='min',
    monitor_modality_grad=False,
    aca: Optional[AdaptiveClassifierAssignment] = None,
    log_callback: Optional[Callable[[int, Dict, Optional[Dict]], None]] = None,
):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    log_ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(results_dir, f'training_{log_ts}.log')
    with open(log_path, 'w') as log_f:
        pass

    history = {}

    for epoch in range(1, num_epoch + 1):
        printlog('Epoch {0} / {1}'.format(epoch, num_epoch))

        train_step_runner = StepRunner(
            net=net,
            stage='train',
            loss_fn=loss_fn,
            metrics_dict=None,
            optimizer=optimizer,
            device=device,
            monitor_modality_grad=monitor_modality_grad,
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_log = train_epoch_runner(train_data)

        for name, metric in train_log.items():
            history[name] = history.get(name, []) + [metric]

        task_loss = train_log['train_loss']
        did_val = (epoch % val_freq == 0) and val_data is not None
        val_log = None
        aca_added = None
        n_heads = net.get_num_heads() if hasattr(net, 'get_num_heads') else None

        if aca is not None and aca.should_check(epoch):
            scores = aca.compute_confidence_scores(net, train_data, device)
            added = aca.assign(net, scores)
            if added is not None:
                aca_added = added
                current_lr = optimizer.param_groups[0]['lr']
                new_head = net.modality_heads[added][-1]
                optimizer.add_param_group({'params': new_head.parameters(), 'lr': current_lr})
                n_heads = net.get_num_heads()
                print(
                    f'Epoch {epoch}: ACA added head to modality {added}, n_heads: {n_heads}',
                    file=sys.stderr,
                )

        if did_val:
            val_step_runner = StepRunner(
                net=net,
                stage='val',
                loss_fn=loss_fn,
                metrics_dict=deepcopy(metrics_dict),
                device=device,
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_log = val_epoch_runner(val_data)
            val_log['val_epoch'] = epoch
            for name, metric in val_log.items():
                history[name] = history.get(name, []) + [metric]

        line = _fmt_log_line_boosted(
            epoch, task_loss, train_log, did_val,
            val_log if did_val else None,
            aca_added=aca_added,
            n_heads=n_heads,
        )
        with open(log_path, 'a') as log_f:
            log_f.write(line + '\n')
            log_f.flush()

        if did_val and val_log is not None and log_callback is not None:
            log_callback(epoch, train_log, val_log)

        if did_val:
            arr_scores = history[monitor]
            best_score_idx = np.argmax(arr_scores) if eval_mode == 'max' else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_bst.pt'))
                print(
                    '<<<<<< reach best {0} : {1} >>>>>>'.format(monitor, arr_scores[best_score_idx]),
                    file=sys.stderr,
                )
            if apply_early_stopping and len(arr_scores) - best_score_idx > patience:
                print(
                    '<<<<<< {} without improvement in {} turns, early stopping >>>>>>'.format(
                        monitor, patience
                    ),
                    file=sys.stderr,
                )
                break

            with open(os.path.join(results_dir, 'hist.json'), 'w') as outfile:
                json.dump(history, outfile)

    torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_final.pt'))
    print(history)
    return history
