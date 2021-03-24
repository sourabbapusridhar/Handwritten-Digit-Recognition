# Implementation of the class BaseDataLoader

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """
    """
    def __init__(self, dataset, batchSize, shuffle, validationSplit, numWorkers, collateFunction=default_collate):
        """
        """
        pass

    def _split_sampler(self, split):
        """
        """
        pass

    def split_validaion(self):
        """
        """
        pass