### base functions
# 1. Not depending on cayley_group or predictor as a class
# 2. useful in many cases

import torch
import time
import numpy  as np

def setup_of_random(seed=None, border=2**32, verbose=0):
    """
    Explicit set of random seed. Works for CPU/GPU

    :param seed           : int or None
    :param border         : int or float

    :return: None, just random seed is set
    """
    if seed is None:
        seed = torch.randint(-border,border+1,(1,)).item()
    if verbose >= 0:
        print(f'\nRandom seed used for experiments: {seed}\n')
    torch.manual_seed(seed)
    return seed

################################################################################################################################################################################################################################################################################
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
################################################################################################################################################################################################################################################################################   
def isin_via_searchsorted(elements,test_elements_sorted):  
    """
    Must be faster than isin(assume_unique=True), but needs sort instead of deduplication (it's also can be using sort)
    I've tested it and now I'm pretty sure it works correctly

    :param elements             : torch.tensor
    :param test_elements_sorted : torch.tensor - must be sorted

    :return: boolean tensor the size of elements
    """
    ts = torch.searchsorted(test_elements_sorted,elements)
    ts[ts>=len(test_elements_sorted)] = len(test_elements_sorted)-1
    return (test_elements_sorted[ts] == elements)

################################################################################################################################################################################################################################################################################   
def isin_via_searchsorted_w_sort_inside(elements,test_elements_sorted):
    """
    same as isin_via_searchsorted but test_elements_sorted is sorted inside

    :param elements             : torch.tensor
    :param test_elements_sorted : torch.tensor

    :return: boolean tensor the size of elements
    """
    test_elements_sorted,_ = test_elements_sorted.sort(stable=True)
    ts = torch.searchsorted(test_elements_sorted,elements)
    ts[ts>=len(test_elements_sorted)] = len(test_elements_sorted)-1
    return (test_elements_sorted[ts] == elements)

################################################################################################################################################################################################################################################################################
def get_neighbors_plain(states, moves):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    return torch.gather(states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)), 
                        2, 
                        moves .unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1))).flatten(end_dim=1) # added flatten to the end, because we always add it

################################################################################################################################################################################################################################################################################
# def get_neighbors(states, moves, chunk_size=2**14):
#     """
#     Some torch magic to calculate all new states which can be obtained from states by moves
#     """
#     if chunk_size>0:
#         #return torch.vstack([get_neighbors_plain(states[i:i+chunk_size, :], moves) for i in range(0,len(states),chunk_size)])
#         result = torch.zeros(states.shape[0]*moves.shape[0], states.shape[1], dtype=states.dtype, device=states.device)
#         for i in range(0,len(states),chunk_size):
#             result[i*moves.shape[0]:(i+chunk_size)*moves.shape[0], :] = get_neighbors_plain(torch.narrow(states, 0, i, chunk_size), moves) #states[i:i+chunk_size, :]
#         return result
#     return get_neighbors_plain(states, moves)

################################################################################################################################################################################################################################################################################
def get_neighbors2(states, moves, chunking_thres=2**18):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    s_sh = states.shape[0]
    if s_sh > chunking_thres:
        result = torch.zeros(s_sh*moves.shape[0], states.shape[1], dtype=states.dtype, device=states.device)
        for i in range(0,moves.shape[0]):
            result[i*s_sh:(i+1)*s_sh, :] = get_neighbors_plain(states, torch.narrow(moves, 0, i, 1))#moves[i:i+1,:])#torch.narrow(moves, 0, i, 1))#states[:,moves[i,:]]
        return result
    return get_neighbors_plain(states, moves)

################################################################################################################################################################################################################################################################################

def pred_d(states, state_destination, model, device, n_gens):
    """
    batched loading to model
    """
    states = states.to(torch.int64)
    pred_v, pred_p = batch_processVP(model, states, n_gens, device, 4096)
    mask_finish = (states==state_destination).all(dim=1)
    pred_v[mask_finish] = 0
    if device == 'cpu':
        torch.clip(pred_v, 0, torch.inf).cpu(), pred_p.cpu(), mask_finish.any().cpu()
    return torch.clip(pred_v, 0, torch.inf), pred_p, mask_finish.any()

################################################################################################################################################################################################################################################################################
def batch_processVP(model, data, n_gens, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    n_samples = data.size(0)
    outputs_v = torch.zeros((n_samples,      ), dtype=torch.float64, device=device)
    outputs_p = torch.zeros((n_samples,n_gens), dtype=torch.float64, device=device)

    # Process each batch
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch = data[start:end].to(device)
        with torch.no_grad():
            batch_output_v, batch_output_p = model(batch)
        
        # Store the output
        outputs_v[start:end] = batch_output_v
        outputs_p[start:end] = batch_output_p

    return outputs_v, outputs_p

################################################################################################################################################################################################################################################################################
def model_predict_in_batches(models_or_heuristics, data, batch_size=2**16, device=None, **kwargs):
    n_states_all = data.shape[0]
    estimations  = torch.zeros( n_states_all, device = data.device if device is None else device, dtype = torch.float32 )
    state_destination = kwargs.get('state_destination',None)
    for start_loc  in range(0,n_states_all,batch_size ):
        end_loc = min( [start_loc + batch_size, n_states_all ] )
        with torch.no_grad():
            estimations[start_loc:end_loc] = models_or_heuristics( data[start_loc:end_loc] ).reshape(-1) if state_destination is None else models_or_heuristics( data[start_loc:end_loc], state_destination ).reshape(-1)
    return estimations

################################################################################################################################################################################################################################################################################
def check_stagnation(states_log, i, p=0.8):
    """
    If 2 cols of table are mostly in 2 other cols - this means a process is stagnating and be better restarted

    :param states_log: torch.tensor
    :param i         : int / step of outer beam search
    :param p         : float / percent of intersection to be counted as a stagnation
    
    :return: True if stagnating, False otherwise
    """

    a_iteration   = torch.hstack([states_log[(i-1)%4,:].flatten(),states_log[(i-2)%4,:].flatten()])
    a_iteration   = a_iteration[a_iteration!=0]

    p_iteration   = torch.hstack([states_log[(i-3)%4,:].flatten(),states_log[(i-4)%4,:].flatten()])
    p_iteration,_ = p_iteration[p_iteration!=0].sort(stable=True)

    return ((2*torch.sum(isin_via_searchsorted(a_iteration, p_iteration)))/(states_log[states_log!=0].shape[0]) >= p) or (len(a_iteration) == 0)
    
################################################################################################################################################################################################################################################################################
def hamming_dist(stts, state_destination, chunking_thres=2**18):
    if stts.shape[0]>chunking_thres:
        return torch.hstack([torch.sum( (stts_chunk != state_destination) , dim = 1) for stts_chunk in torch.tensor_split(stts, stts.shape[0]//chunking_thres+1)])
    return torch.sum( (stts != state_destination) , dim = 1)

################################################################################################################################################################################################################################################################################
def hstack_if_not_None(l, r):
    return torch.hstack([l,r]) if l is not None else r

################################################################################################################################################################################################################################################################################
def vstack_if_not_None(l, r):
    return torch.vstack([l,r]) if l is not None else r

################################################################################################################################################################################################################################################################################
def tensor_split_preprocessed(M, thres):
    if M.shape[0]>thres:
        chunks = (M.shape[0]//thres) + 1 - (M.shape[0]%thres == 0)
        return torch.tensor_split(M, chunks)
    return [M,]
            
################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################
