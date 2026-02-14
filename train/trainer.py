"""Training loop for Baseline Late Fusion: no sample weights, single-stage end-to-end."""
import os
import sys
import numpy as np
import datetime
from tqdm import tqdm
import torch
from copy import deepcopy
import json
from torch.utils.data import DataLoader


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n' + '==========' * 8 + '%s' % nowtime)
    print(str(info) + '\n')


def _collect_modality_grad_stats(net, n_experts=4, eps=1e-8):
    """Collect gradient norms and ratios from first-layer encoder outputs. Assumes hooks were run and backward done."""
    acts = [None] * n_experts
    handles = []

    def make_hook(idx):
        def hook(module, input, output):
            output.retain_grad()
            acts[idx] = output
        return hook

    if not hasattr(net, 'expert_ls'):
        return None
    for i in range(min(n_experts, len(net.expert_ls))):
        expert = net.expert_ls[i]
        if not hasattr(expert, 'encoder_stages') or len(expert.encoder_stages) == 0:
            return None
        h = expert.encoder_stages[0].register_forward_hook(make_hook(i))
        handles.append(h)

    return acts, handles


class StepRunner:
    def __init__(self, net, loss_fn,
                 stage='train', metrics_dict=None,
                 optimizer=None,
                 device='cuda:0',
                 monitor_modality_grad=False,
                 ):
        self.net = net
        self.loss_fn, self.metrics_dict, self.stage = loss_fn, metrics_dict, stage
        self.optimizer = optimizer
        self.device = device
        self.monitor_modality_grad = monitor_modality_grad

    def step(self, batch):
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        grad_stats = None
        if self.stage == 'train' and self.monitor_modality_grad and hasattr(self.net, 'expert_ls'):
            n_experts = len(self.net.expert_ls)
            hook_result = _collect_modality_grad_stats(self.net, n_experts)
            if hook_result is not None:
                acts, handles = hook_result
                try:
                    preds = self.net(features)
                    loss = self.loss_fn(preds, labels)
                    loss.backward()

                    eps = 1e-8
                    norms = []
                    for i in range(n_experts):
                        t = acts[i]
                        if t is not None and t.grad is not None:
                            norms.append(t.grad.norm().item())
                        else:
                            norms.append(0.0)
                    total = sum(norms) + eps
                    ratios = [n / total for n in norms]
                    balance = min(norms) / (max(norms) + eps) if max(norms) > 0 else 0.0

                    grad_stats = {
                        'grad_norm_0': norms[0], 'grad_norm_1': norms[1], 'grad_norm_2': norms[2], 'grad_norm_3': norms[3],
                        'grad_ratio_0': ratios[0], 'grad_ratio_1': ratios[1], 'grad_ratio_2': ratios[2], 'grad_ratio_3': ratios[3],
                        'grad_balance': balance,
                    }
                finally:
                    for h in handles:
                        h.remove()
                self.optimizer.step()
                self.optimizer.zero_grad()
                return loss.item(), len(labels), grad_stats

        preds = self.net(features)
        loss = self.loss_fn(preds, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), len(labels), grad_stats

    def step_infer(self, batch):
        features = batch['img'].to(self.device)
        labels = batch['label'].to(self.device)

        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        step_metrics = {
            self.stage + '_' + name: metric_fn(preds, labels).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics, len(labels)

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
        else:
            return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner: StepRunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def _train_handler(self):
        cur_loss = 0
        cur_num_sample = 0
        cur_grad_stats = {}
        loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout)

        for step, batch in loop:
            result = self.steprunner(batch)
            step_loss, batch_size = result[0], result[1]
            grad_stats = result[2] if len(result) == 3 else None

            cur_loss = (cur_num_sample * cur_loss + batch_size * step_loss) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size

            if grad_stats is not None:
                for k, v in grad_stats.items():
                    cur_grad_stats[k] = (cur_num_sample - batch_size) * cur_grad_stats.get(k, 0) + batch_size * v
                for k in cur_grad_stats:
                    cur_grad_stats[k] = cur_grad_stats[k] / cur_num_sample

            cur_log = dict({self.stage + '_loss': cur_loss})
            for k, v in cur_grad_stats.items():
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
                    cur_metrics[metric_name] = (cur_num_sample * cur_metrics[metric_name] + batch_size * metric_value) / (cur_num_sample + batch_size)
            cur_num_sample += batch_size
            cur_log = dict({self.stage + '_loss': cur_loss}, **cur_metrics)
            loop.set_postfix(**cur_log)
        return cur_log

    def __call__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        if self.stage == 'train':
            cur_log = self._train_handler()
        else:
            cur_log = self._eval_handler()
        return cur_log


def _fmt_log_line_simmlm(epoch, task_loss, train_log, did_val, val_log=None):
    """Format one epoch line in SimMLM style: [timestamp] Epoch N | grad_encoder: ... | encoder_ratio: ...% | task_loss: ... [| val_... ]"""
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
    parts.append(f'task_loss: {task_loss:.4f}')
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


def train_model(net, optimizer, loss_fn, metrics_dict, train_data, val_data, val_freq=1,
                num_epoch=100, results_dir='results/0/', device='cuda:0',
                apply_early_stopping=False, patience=5, monitor='val_loss', eval_mode='min',
                monitor_modality_grad=False):
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
            net=net, stage='train',
            loss_fn=loss_fn, metrics_dict=None,
            optimizer=optimizer, device=device,
            monitor_modality_grad=monitor_modality_grad,
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_log = train_epoch_runner(train_data)

        for name, metric in train_log.items():
            history[name] = history.get(name, []) + [metric]

        task_loss = train_log['train_loss']
        did_val = (epoch % val_freq == 0) and val_data is not None
        val_log = None

        if did_val:
            val_step_runner = StepRunner(
                net=net, stage='val', loss_fn=loss_fn,
                metrics_dict=deepcopy(metrics_dict), device=device
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_log = val_epoch_runner(val_data)
            val_log['val_epoch'] = epoch
            for name, metric in val_log.items():
                history[name] = history.get(name, []) + [metric]

        # Write one line to training log (SimMLM-style: [timestamp] Epoch N | ... )
        line = _fmt_log_line_simmlm(epoch, task_loss, train_log, did_val, val_log if did_val else None)
        with open(log_path, 'a') as log_f:
            log_f.write(line + '\n')
            log_f.flush()

        if did_val:
            arr_scores = history[monitor]
            best_score_idx = np.argmax(arr_scores) if eval_mode == 'max' else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_bst.pt'))
                print('<<<<<< reach best {0} : {1} >>>>>>'.format(monitor, arr_scores[best_score_idx]), file=sys.stderr)
            if apply_early_stopping and len(arr_scores) - best_score_idx > patience:
                print('<<<<<< {} without improvement in {} turns of validations, early stopping >>>>>>'.format(monitor, patience), file=sys.stderr)
                break

            with open(os.path.join(results_dir, 'hist.json'), 'w') as outfile:
                json.dump(history, outfile)

    torch.save(net.state_dict(), os.path.join(results_dir, 'ckpt_final.pt'))
    print(history)
    return history
