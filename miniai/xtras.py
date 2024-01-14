from time import time
from copy import copy
from torcheval.metrics import Mean
import torchvision
from torch import tensor, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import fastcore.all as fc
import matplotlib.pyplot as plt
from fastprogress import progress_bar,master_bar

from .datasets import *
from .learner import *
from .activations import *

# Code from fastcore https://github.com/fastai/fastprogress/blob/master/fastprogress/core.py#L7
def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h,m,s = t//3600, (t//60)%60, t%60
    if h!= 0: return f'{h}:{m:02d}:{s:02d}'
    else:     return f'{m:02d}:{s:02d}'

class MetricsCB(Callback):
    """Added time for the original."""
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, x): print(x)

    def before_fit(self, learn): learn.metrics = self

    def before_epoch(self, learn):
        for m in self.all_metrics.values(): m.reset()
        self.start_time = time()

    def after_epoch(self, learn):
        log = {k: f'{v.compute():.3f}' for k, v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = learn.model.training
        log['time'] = format_time(time() - self.start_time)
        self._log(log)

    def after_batch(self, learn):
        x, y = learn.batch
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), to_cpu(y))
        self.loss.update(to_cpu(learn.loss), weight=len(x))

def get_dls(bs=1024, seed=42):
    """Grab MNIST fasion data using pytorch dataset."""
    set_seed(seed)
    bs = bs
    xmean, xstd = (tensor(0.29), tensor(0.35))
    def batch_tfm(img): return (ToTensor()(img) - xmean) / xstd
    trn_ds = datasets.FashionMNIST(
        root=".", train=True, download=True, transform=batch_tfm
    )
    val_ds = datasets.FashionMNIST(
        root=".", train=False, download=True, transform=batch_tfm
    )
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True)
    return DataLoaders(trn_dl, val_dl)

class ActivationStats(HooksCallback):
    """Add after fit to plot statistics."""
    def __init__(self, mod_filter=fc.noop, plot=False):
        super().__init__(append_stats, mod_filter)
        self.plot = plot

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)

    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))

    def after_fit(self, learn):
        if self.plot: self.plot_stats(figsize=(8, 3))
        super().after_fit(learn)

class MyNorm(nn.Module):
    """Simple like layernorm but taking mean/std like batch norm."""
    def __init__(self, dummy, eps = 1e-4):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.))
        self.add = nn.Parameter(tensor(0.))

    def forward(self, x):
        mean = x.mean((0,2,3), keepdim=True)  # NCHW
        std  = x.std((0,2,3), keepdim=True)
        x = (x - mean) / (std+self.eps).sqrt()
        return (x * self.mult) + self.add

class ProgressCB(Callback):
    """
    Plot frequency options:
    batch: every batch
    epoch: every epoch
    fit: first two epochs and after fit
    """
    order = MetricsCB.order+1
    def __init__(self, plot=False, update_freq='fit', epoch_nums=2): fc.store_attr()
    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)
        
    def _update_graph(self, learn, valid):
        if not self.losses: return
        n = int(valid)
        self.mbar.update_graph(
            [[fc.L.range(self.losses), self.losses],
            [fc.L.range(learn.epoch+n).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]])

    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses and self.update_freq=='batch': self._update_graph(learn, valid=False)
    
    def after_epoch(self, learn): 
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'): 
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                if self.update_freq=='epoch' or (self.update_freq=='fit' and learn.epoch<self.epoch_nums): 
                    self._update_graph(learn, valid=True)

    def after_fit(self, learn):
        if self.plot and self.update_freq == 'fit': self._update_graph(learn, valid=True)
