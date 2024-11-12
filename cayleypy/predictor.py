### class for model execution
# any object having method "predict" or "__call__" (this means instance can be used as a function name) could be used as a predictor
# torch.jit.load usage is a legacy and would be deprecated in a future

import torch
import time
import numpy  as np

from .utils import *

class Predictor:
    """
    Unified class to call model / metric and do not put model code into beam search code
    """
    def __init__(self, models_or_heuristics, need_state_destination = False, batching = True, device = 'Auto'):
        self.predict = None
        # it's better for speed to store all data in the same device
        if device == 'Auto':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.models_or_heuristics   = models_or_heuristics
        self.batching               = batching
        self.need_state_destination = False
            
        if  models_or_heuristics == "Hamming":          # hamming needs to know state destination and all others do not
            self.predict                = hamming_dist
            self.need_state_destination = True
        elif isinstance(models_or_heuristics, str) and models_or_heuristics.startswith("../../kaggle/input"):
            self.predict = torch.jit.load(models_or_heuristics, map_location=self.device)
            self.predict.eval()
            self.predict.to(self.device)
        elif isinstance(models_or_heuristics, torch.nn.Module):
            self.predict = models_or_heuristics
            self.predict.eval()
            self.predict.to(self.device)
        elif hasattr(models_or_heuristics, 'predict'):
            self.predict = models_or_heuristics.predict
        elif hasattr(models_or_heuristics, '__call__'):
            self.predict = models_or_heuristics
        else:
            raise ValueError(f'Unable to understand how to call {models_or_heuristics}')
            
    def __call__(self, data, group_instance):
        state_destination = group_instance.state_destination if self.need_state_destination else None
        if self.batching:
            res = model_predict_in_batches(data=data, models_or_heuristics=self.predict, state_destination=state_destination, device=self.device)
        else:
            res = self.predict(data, state_destination=state_destination)
        return res