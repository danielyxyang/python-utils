__all__ = [
    "seed_everything",
    "seed_dataloader",
    "OutputChecker",
    "OUTPUT_CHECKER",
]

import os
import random
import contextlib
import logging
import pickle

import numpy as np
try:
    import torch
    import torch.nn as nn
    _IMPORTED_TORCH = True
except ImportError:
    _IMPORTED_TORCH = False


logger = logging.getLogger(__name__)

def seed_everything(seed=None, deterministic=False):
    """Set seeds and ensures usage of deterministic algorithms.
    
    Args:
        seed (int, optional): The seed set for each dataloader worker. Defaults
            to None.
        deterministic (bool, optional): Flag whether algorithms should be as
            deterministic as possible. Defaults to False.
    """
    # https://pytorch.org/docs/stable/notes/randomness.html
    # seed random number generators
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if _IMPORTED_TORCH:
            torch.manual_seed(seed)
    # use deterministic algorithms
    if deterministic:
        if _IMPORTED_TORCH:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def _seed_dataloader_worker(worker_id):
    """Set seeds in dataloader workers."""
    if _IMPORTED_TORCH:
        # https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

def seed_dataloader(seed):
    """Return arguments to set the seed in dataloader workers.
    
    Args:
        seed (int): The seed set for each dataloader worker.

    Returns:
        dict: Keyword arguments to be provided to dataloader constructor.
    """
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    if _IMPORTED_TORCH:
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            return dict(
                worker_init_fn=_seed_dataloader_worker,
                generator=g,
            )
        else:
            return dict()


class OutputChecker:
    """Class for checking reproducibility of function outputs."""
    COLLECT = "COLLECT"
    VERIFY = "VERIFY"

    def __init__(self):
        self.phase = None
        self.scopes = []
        self.outputs = {}
    
    def reset(self):
        self.phase = None
        self.scopes.clear()
        self.outputs.clear()

    @contextlib.contextmanager
    def collect(self, path=None):
        try:
            # start collection of outputs
            logger.info("Collecting outputs...")
            self.phase = OutputChecker.COLLECT
            seed_everything(0)
            yield
        finally:
            self.phase = None
            # store collected outputs to file
            if path is not None:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(self.outputs, f)
                logger.info(f"Collected outputs saved to {path}.")
    
    @contextlib.contextmanager
    def verify(self, path=None):
        try:
            # load previously collected outputs from file
            if path is not None:
                with open(path, "rb") as f:
                    self.outputs = pickle.load(f)
                logger.info(f"Loaded collected outputs from {path}.")
            # compute max length of name for pretty printing
            self.names_max_len = max(len(name) for name in self.outputs)
            # start verification of outputs
            logger.info("Verifying outputs...")
            self.phase = OutputChecker.VERIFY
            seed_everything(0)
            yield
        finally:
            self.phase = None

    @contextlib.contextmanager
    def scope(self, name):
        try:
            self.scopes.append(name)
            yield
        finally:
            self.scopes.pop()

    def __call__(self, name, obj, disable=False):
        if _IMPORTED_TORCH and isinstance(obj, nn.Module):
            obj.register_forward_hook(lambda module, args, output: self(name, lambda: output))
            return obj
        elif callable(obj):
            with self.scope(name):
                out = obj()
            # collect or verify output according to phase and if not disabled
            if self.phase is not None and not disable:
                name = ".".join(self.scopes + [name])
                if self.phase == OutputChecker.COLLECT:
                    if name not in self.outputs:
                        # collect output
                        self.outputs[name] = out
                        # return output
                        return out
                    else:
                        logger.warn(f"{name} has already been collected.")
                elif self.phase == OutputChecker.VERIFY:
                    if name in self.outputs:
                        # verify output
                        mean, max, sum = OutputChecker.diff(out, self.outputs[name])
                        print(f"{name:{self.names_max_len}} {mean:9.4g} {max:9.4g} {sum:9.4g}")
                        # return previously collected output to prevent accumulation of differences
                        out_prev = OutputChecker._update_tensors(self.outputs[name], out)
                        return out_prev
                    else:
                        logger.warn(f"{name} has not been collected.")
                else:
                    logger.warn(f"Unknown OutputChecker phase {self.phase}.")
            return out
        else:
            logger.warn(f"Object of type {type(obj)} can not be checked, since it is not callable.")
            return obj
    
    @staticmethod
    def _update_tensors(out_prev, out):
        if type(out_prev) is not type(out):
            logger.warn(f"Output with different types {type(out_prev)} and {type(out)}.")
            return out_prev
        
        if _IMPORTED_TORCH and isinstance(out_prev, torch.Tensor):
            # copy previous values into current tensor to preserve computation graph
            out.data.copy_(out_prev)
            return out
        elif isinstance(out_prev, (list, tuple)):
            if len(out_prev) == len(out):
                list_or_tuple = type(out_prev)
                return list_or_tuple(map(OutputChecker._update_tensors, out_prev, out))
            else:
                logger.warn(f"Output of type {type(out_prev)} with different length {len(out_prev)} and {len(out)}.")
                return out_prev
        elif isinstance(out_prev, dict):
            if out_prev.keys() == out.keys():
                return {key: OutputChecker._update_tensors(out_prev[key], out[key]) for key in out_prev}
            else:
                logger.warn(f"Output of type {type(out_prev)} with different set of keys {list(out_prev.keys())} and {list(out.keys())}.")
                return out_prev
        else:
            return out_prev
    
    @staticmethod
    def diff(out1, out2):
        if type(out1) is not type(out2):
            logger.warn(f"Output with different types {type(out1)} and {type(out2)}.")
            return 0, 0, 0
        
        if isinstance(out1, np.ndarray):
            diffs = np.abs(out1 - out2)
            return np.mean(diffs), np.max(diffs), np.sum(diffs)
        elif _IMPORTED_TORCH and isinstance(out1, torch.Tensor):
            diffs = np.abs(out1.detach().cpu().numpy() - out2.detach().cpu().numpy())
            return np.mean(diffs), np.max(diffs), np.sum(diffs)
        elif isinstance(out1, (list, tuple)):
            if len(out1) == len(out2):
                means, maxs, sums = zip(*[OutputChecker.diff(o1, o2) for o1, o2 in zip(out1, out2)])
                return np.mean(means), np.max(maxs), np.sum(sums)
            else:
                logger.warn(f"Output of type {type(out1)} with different length {len(out1)} and {len(out2)}.")
                return 0, 0, 0
        elif isinstance(out1, dict):
            if out1.keys() == out2.keys():
                means, maxs, sums = zip(*[OutputChecker.diff(out1[key], out2[key]) for key in out1])
                return np.mean(means), np.max(maxs), np.sum(sums)
            else:
                logger.warn(f"Output of type {type(out1)} with different set of keys {list(out1.keys())} and {list(out2.keys())}.")
                return 0, 0, 0
        else:
            logger.warn(f"Output type {type(out1)} not supported.")
            return 0, 0, 0

OUTPUT_CHECKER = OutputChecker()
