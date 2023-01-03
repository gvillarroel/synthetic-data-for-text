
from copy import deepcopy
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path



def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda'), checkpoint=None):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.checkpoint = checkpoint
        if self.checkpoint and not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        best_loop = 10

        if self.checkpoint and os.path.exists(os.path.join(self.checkpoint, 'best_model.pt')):
            self.diffusion.load_state_dict(torch.load(os.path.join(self.checkpoint, 'best_model.pt')))
            self.ema_model.load_state_dict(torch.load(os.path.join(self.checkpoint, 'best_model_ema.pt')))

        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
                if step == 0:
                    best_loop = mloss + gloss
                if best_loop > mloss + gloss and step > 0:
                    last_best = best_loop
                    best_loop = mloss + gloss
                    if self.checkpoint:
                        torch.save(self.diffusion.state_dict(), os.path.join(self.checkpoint, 'best_model.pt'))
                        torch.save(self.ema_model.state_dict(), os.path.join(self.checkpoint, 'best_model_ema.pt'))
                        torch.save(self.diffusion.state_dict(), os.path.join(self.checkpoint, f'best_model-{best_loop}.pt'))
                        torch.save(self.ema_model.state_dict(), os.path.join(self.checkpoint, f'best_model_ema-{best_loop}.pt'))
                        if os.path.exists(os.path.join(self.checkpoint, f'best_model-{last_best}.pt')):
                            os.remove(os.path.join(self.checkpoint, f'best_model-{last_best}.pt'))
                        if os.path.exists(os.path.join(self.checkpoint, f'best_model_ema-{last_best}.pt')):
                            os.remove(os.path.join(self.checkpoint, f'best_model_ema-{last_best}.pt'))
                if self.checkpoint and np.isnan(mloss + gloss):
                    print(f"exit {self.checkpoint}")
                    Path(f'{self.checkpoint}/exit').touch()
                    exit(1)
                    print("rollback")
                    self.diffusion.load_state_dict(torch.load(os.path.join(self.checkpoint, 'best_model.pt')))
                    self.ema_model.load_state_dict(torch.load(os.path.join(self.checkpoint, 'best_model_ema.pt')))
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            

            step += 1