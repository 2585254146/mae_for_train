import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Optional
import numpy as np
import torch
import logging


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

def set_seed(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False

class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Keyword Arguments:
        directory {str} -- Path to save logs at (default: {''})
        name {str} -- Name of the file to save the logs in (default: {'logs.txt'})
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    logfile = directory + name + ".txt"
    print(f"Logging outputs at {logfile}")

    log = open(logfile, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger



def count_parameters(model, all=False):
    """
    Count the parameters of a model
    
    Arguments:
        model {torch module} -- Model to count the parameters of
    
    Keyword Arguments:
        all {bool} -- Whether to include not trainable parameters in the sum (default: {False})
    
    Returns:
        int -- Number of parameters
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)









