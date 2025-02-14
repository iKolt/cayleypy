 ### main class for beam search

### TODO
# после инциализации , какие функции 

# 1) Основное - для данного элемента - построить путь , но надо продумать интрефейс - чтобы модель не училась 2 дня , в простых случаях - мы можем без модели - бфс, хемминг и тд 

# На будущее:
# 2) Выдать граф (хотя бы кусок) - спрас матрица
# 3) Нарисовать граф
# 4) Подсчитать рост хотя б первые члены
# 5) Рендом волки генерить
# 6) Рендом волки рисовать и  анализировать 
# 7) Оценить диаметр - приудмать как
# 8) Эмбедеинг
# 9)  спектр Лапласиана и вокруг него

# Более техническое - полезное уже сейчас
# 10) выдавать окрестности точки 
# .....

###
# - смотреть в сторону A* (N шагов должны приближать на qN)
# - рестарт не с начала, а с хороших точек на пути
# - усреднять эстимейт на всех соседях для лучшего подсчёта энергии
# - добавлять в нонбэктрек все точки вместе с соседями

# - инкремент спраутинга
# - чистка луча в случайные моменты времени
# - усреднять окрестность хитрее (клипать значения отличающиеся от медианы на ±2)

# - учитывать разброс оценок по соспдям как меру качества оценки

# + у меня есть мысль на ретраях делать инкремент спраутинга (базово +1, но можно и другие функции использовать). На 20й попытке результаты будут сильно отличаться

###
# - бибфс - усреднение окрестности для более точной оценки стейта

###
# 1. Модель тянет к собранному состоянию, а можно добавить метрику, чтобы она "толкала" от стартового состояния
# 2. Если мы застряли в лок.минимуме, можно поменять objective и временно (N шагов или до порога) двигаться к максимумам луча, а потом снова к минимумам

import torch
import time
import numpy  as np

from .utils     import *
from .predictor import *

class CayleyGraph:
    """
    class to encapsulate all of permutation group in one place
    must help keeping reproducibility and dev speed
    """
    ################################################################################################################################################################################################################################################################################
    def __init__(self,
                 
                 generators                ,
                 state_destination = 'Auto',
                 
                 vec_hasher        = 'Auto',

                 to_power          = 1.6   ,
                 
                 device            = 'Auto',
                 dtype             = 'Auto',
                 random_seed       = 'Auto' ):
        
        # determine random seed
        if random_seed == 'Auto':
            setup_of_random()
        else:
            setup_of_random(random_seed)
        
        # it's better for speed to store all data in the same device
        if device == 'Auto':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        # generators as a list
        if isinstance(generators, list):
            self.list_generators = generators
        elif isinstance(generators, torch.Tensor ):
            self.list_generators = [[q.item() for q in generators[i,:]] for i in range(generators.shape[0] ) ]       
        elif isinstance(generators, np.ndarray ):
            self.list_generators = [list(generators[i,:]) for i in range(generators.shape[0] ) ]
        else:
            print('Unsupported format for "generators"', type(generators), generators)
            raise ValueError('Unsupported format for "generators" ' + str(type(generators)) )    
        
        self.state_size        = len(self.list_generators[0])
        self.n_generators      = len(self.list_generators   )
        
        # dtype 
        if state_destination == 'Auto':
            n_unique_symbols_in_states = self.state_size
        else:
            n_unique_symbols_in_states = len(set( [int(i) for i in state_destination ]  ))
        
        if dtype == 'Auto':
            if n_unique_symbols_in_states <= 256:
                self.dtype = torch.uint8
            else:
                self.dtype = torch.uint16
        elif dtype is not None:
            self.dtype = dtype
                
        self.dtype_generators = torch.int64                                                 # torch.gather raises error if generators aren't long tensor
        #self.dtype_for_hash   = torch.float64 if torch.cuda.is_available() else torch.int64 # less makes collisions much more probable
        # float64 on old models of CUDA - it's a dirty trick but it helps to work really fast
        # possibly can be set to torch.float64 even for CPU and modern GPU
        
        self.dtype_for_hash   = torch.int64 # torch.float64 seems to be erroneus for hashing

        self.dtype_for_dist   = torch.int16
        
        # generators as a tensor
        self.tensor_generators = torch.tensor(   self.list_generators       , device = self.device, dtype = self.dtype_generators, )
        self.generators_dists  = torch.ones_like(self.tensor_generators[:,0], device = self.device, dtype = self.dtype_generators, )
        
        if state_destination == 'Auto':
            self.state_destination = torch.arange( self.state_size, device=self.device, dtype = self.dtype).reshape(-1,self.state_size)
        elif isinstance(state_destination, torch.Tensor ):
            self.state_destination =  state_destination.to(self.device).to(self.dtype).reshape(-1,self.state_size)
        else:
            self.state_destination = torch.tensor( state_destination, device=self.device, dtype = self.dtype).reshape(-1,self.state_size)
            
        
        if vec_hasher == 'Auto':
            # Hash vector generation
            max_int =  int( (2**62) )
            self.vec_hasher = torch.randint(-max_int, max_int+1, size=(self.state_size,), device=self.device, dtype=self.dtype_for_hash) #
        elif not isinstance( vec_hasher , torch.Tensor):
            self.vec_hasher = torch.tensor( vec_hasher , device=self.device, dtype=self.dtype_for_hash )
        else:
            self.vec_hasher = vec_hasher.to(self.device).to(self.dtype_for_hash)
            
        self.predictor = None
        
        self.define_make_hashes()

        self.manhatten_moves_matrix_count(to_power=to_power)
        
    ################################################################################################################################################################################################################################################################################
    
    # bit of setup to get the fastest make_hashes - now it's possible always make_hashes_cpu_and_modern_gpu because of using float64 for self.dtype_for_hash
    
    def define_make_hashes(self):
        try:
            _ = self.make_hashes_cpu_and_modern_gpu( torch.vstack([self.state_destination,
                                                                   self.state_destination,]) )
            self.make_hashes = self.make_hashes_cpu_and_modern_gpu
        except Exception as e:
            self.make_hashes = self.make_hashes_older_gpu
    
    def make_hashes_cpu_and_modern_gpu(self, states: torch.Tensor, chunk_size_thres=2**18):
        #if states.shape[0]>chunk_size_thres:
        #    chunk_size = states.shape[0]//8+1 # 8 because we're creating about 8bytes for each row of input
        #    result     = torch.zeros_like(states[:,0], dtype=self.dtype_for_hash, device=self.device)
        #    for i in range(0,states.shape[0],chunk_size):
        #        _sz = min(i+chunk_size, states.shape[0])-i
        #        result[i:i+_sz] = torch.narrow(states, 0, i, _sz).to(self.dtype_for_hash) @ self.vec_hasher.mT
        #else:
        #    result = states.to(self.dtype_for_hash) @ self.vec_hasher.mT
        #return result
        return states.to(self.dtype_for_hash) @ self.vec_hasher.mT if states.shape[0]<=chunk_size_thres else torch.hstack([(z.to(self.dtype_for_hash) @ self.vec_hasher.reshape((-1,1))).flatten() for z in torch.tensor_split(states,8)])
    
    def make_hashes_older_gpu(self, states: torch.Tensor, chunk_size_thres=2**18):
        #if states.shape[0]>chunk_size_thres:
        #    chunk_size = states.shape[0]//8+1 # 8 because we're creating about 8bytes for each row of input
        #    result     = torch.zeros_like(states[:,0], dtype=self.dtype_for_hash, device=self.device)
        #    for i in range(0,states.shape[0],chunk_size):
        #        _sz = min(i+chunk_size, states.shape[0])-i
        #        result[i:i+_sz] = torch.sum( torch.narrow(states, 0, i, _sz).to(self.dtype_for_hash) * self.vec_hasher, dim=1)
        #else:
        #    result = torch.sum( states.to(self.dtype_for_hash) * self.vec_hasher, dim=1)
        #return result 
        return torch.sum( states * self.vec_hasher, dim=1) if states.shape[0]<=chunk_size_thres else torch.hstack([torch.sum( z * self.vec_hasher, dim=1) for z in torch.tensor_split(states,8)])
                                                            # Compute hashes. 
                                                            # It is same as matrix product torch.matmul(hash_vec , states ) 
                                                            # but pay attention: such code work with GPU for integers 
                                                            # While torch.matmul - does not work for GPU for integer data types, 
                                                            # since old GPU hardware (before 2020: P100, T4) does not support integer matrix multiplication
            
    ################################################################################################################################################################################################################################################################################
    def get_unique_states_2( self, states: torch.Tensor, flag_already_hashed : bool = False ) -> torch.Tensor:
        '''
        Return matrix with unique rows for input matrix "states" 
        I.e. duplicate rows are dropped.
        For fast implementation: we use hashing via scalar/dot product.
        Note: output order of rows is different from the original. 
        '''
        # Note: that implementation is 30 times faster than torch.unique(states, dim = 0) - because we use hashes  (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)
        # Note: torch.unique does not support returning of indices of unique element so we cannot use it 
        # That is in contrast to numpy.unique which supports - set: return_index = True 

        # Hashing rows of states matrix: 
        hashed = states if flag_already_hashed else self.make_hashes(states)

        # sort
        hashed_sorted, idx = torch.sort(hashed, stable=True)

        # Mask initialization
        mask = torch.ones(hashed_sorted.size(0), dtype=torch.bool, device=self.device)

        # Mask main part:
        if hashed_sorted.size(0) > 1:
            mask[1:] = (hashed_sorted[1:] != hashed_sorted[:-1])

        # Update index
        IX1 = idx[mask]

        return states[IX1], hashed[IX1], IX1
    
    ################################################################################################################################################################################################################################################################################
    def get_neighbors_chunked_iterator( self, states, moves, states_attributes=None, moves_attributes=None, chunking_thres=2**18 ):
        """
        Process states through set of moves in batches.

        :param states           : Tensor of input states / points
        :param moves            : Tensor of possible moves
        :param states_attributes: Tensor
        :param moves_attributes : Tensor
        :param chunk_size_states: Number of samples per batch of states

        :return: neighbours, hashes and attributes in iterator
        """
        if states.shape[0] > chunking_thres//moves.shape[0]:
            for i in range(0,moves.shape[0]):
                neighbs             = get_neighbors_plain(states, torch.narrow(moves, 0, i, 1))#moves[i:i+1,:])#torch.narrow(moves, 0, i, 1)) #moves[i:i+1,:])
                neighbs_hashes, idx = self.make_hashes(neighbs).sort(stable=True) # not doing dedup, but in one chunk all states are unique
                neighbs_attributes  = states_attributes + moves_attributes[i]
                yield neighbs[idx, :], neighbs_hashes, neighbs_attributes[idx]
        else:
            neighbs             = get_neighbors_plain( states, moves )
            neighbs_hashes, idx = self.make_hashes(neighbs).sort(stable=True)
            neighbs_attributes  = states_attributes.repeat(moves.shape[0], 1).T.flatten() +\
                                  moves_attributes .repeat(1,states.shape[0]).T.flatten()
            
            yield neighbs[idx, :], neighbs_hashes, neighbs_attributes[idx]
#         if chunk_size_states == -1:
#             chunk_size_states = states.shape[0]

#         chunk_size_moves = moves.shape[0]

#         for i in range(0, states.shape[0], chunk_size_states ):
#             neighbs = get_neighbors( states[i:i+chunk_size_states, :],
#                                      moves                             )#.flatten(end_dim=1)

#             #_, neighbs_hashes, idx = self.get_unique_states_2(neighbs)
#             neighbs_hashes, idx = self.make_hashes(neighbs).sort(stable=True) # not unique anymore

#             n_act_states        = min(chunk_size_states,states_attributes.shape[0]-i)
#             neighbs_attributes  = states_attributes[i:i+chunk_size_states].repeat(chunk_size_moves, 1).T.flatten() +\
#                                   moves_attributes                        .repeat(1,     n_act_states).T.flatten()

#             yield neighbs[idx, :], neighbs_hashes, neighbs_attributes[idx]

    ################################################################################################################################################################################################################################################################################
    def search_neighbors_for_destination_reach( self, states, moves, stopping_criteria = None, stopping_criteria_hashed = None, states_distances = None, moves_distances = None, already_sorted=True ):#, chunk_size_states=-1,  ):
        """
        Process data through a set of moves in batches to check if we are near destination.

        :param states                  : Tensor of input states / points
        :param moves                   : Tensor of possible moves
        :param stopping_criteria       : Tensor of points we search for intersection with - can be None if stopping_criteria_hashed is not None
        :param stopping_criteria_hashed: Tensor of hashes we search for intersection with - can be None if stopping_criteria        is not None
        :param states_distances        : Tensor
        :param moves_distances         : Tensor
        :param chunk_size_states       : Number of samples per batch of states

        :return: hashes and distances // code can be easily adapted for return of points also
        """

        if stopping_criteria is None and stopping_criteria_hashed is None:
            raise ValueError('stopping_criteria or stopping_criteria_hashed must not be None')

        if stopping_criteria is not None:
            stopping_criteria_hashed = self.make_hashes( stopping_criteria )
        if not already_sorted:
            stopping_criteria_hashed,_ = stopping_criteria_hashed.sort(stable=True)

#         for _, neighbs_hashes, neighbs_dists in self.get_neighbors_chunked_iterator(states,
#                                                                                     moves, 
#                                                                                     states_attributes = states_distances , 
#                                                                                     moves_attributes  = moves_distances  , 
#                                                                                     #chunk_size_states = chunk_size_states
#                                                                                    ):
        chunking_thres = 2**18
        if states.shape[0] > chunking_thres//moves.shape[0]:            
            stop_hashes = []
            stop_dists  = []
            
            for i in range(0,moves.shape[0]):
                neighbs_hashes, idx = self.make_hashes(get_neighbors_plain(states, torch.narrow(moves, 0, i, 1))).sort(stable=True)#moves[i:i+1,:])).sort(stable=True) # not doing dedup, but in one chunk all states are unique#torch.narrow(moves, 0, i, 1)) #moves[i:i+1,:])
                neighbs_dists       = (states_distances + moves_distances[i])[idx]
                
                mask = isin_via_searchsorted(neighbs_hashes.flatten(), stopping_criteria_hashed) # torch.isin(neighbs_hashes.flatten(), stopping_criteria_hashed, assume_unique=True)
                if mask.any():
                    stop_hashes.append(neighbs_hashes[mask])
                    stop_dists .append(neighbs_dists [mask])
            
            if len(stop_hashes)>0:# is not None:
                stop_hashes = torch.hstack(stop_hashes)
                stop_dists  = torch.hstack(stop_dists )
        else:
            neighbs_hashes, idx = self.make_hashes(get_neighbors_plain( states, moves )).sort(stable=True)
            neighbs_dists       = (states_distances.repeat(moves.shape[0], 1).T.flatten() +\
                                   moves_distances .repeat(1,states.shape[0]).T.flatten())[idx]
            mask = isin_via_searchsorted(neighbs_hashes.flatten(), stopping_criteria_hashed)
            stop_hashes = neighbs_hashes[mask]
            stop_dists  = neighbs_dists [mask]
            
        if len(stop_hashes)>0:# is not None:
            
            # if we got to some state by different moves - we must preserve the shortest of paths
            stop_dists, idx = stop_dists.sort(stable=True)
            
            # looks like it's better to deduplicate, not the isin_via_searchsorted (it's useful in beam search that we have only one instance of each hash)                           
            _, stop_hashes, idxs = self.get_unique_states_2(stop_hashes[idx], True)

            return stop_hashes, stop_dists[idxs]

        return stop_hashes, stop_dists

    ################################################################################################################################################################################################################################################################################
    def explode_moves( self, moves, start_states = None, radius=1, initial_distance=1, flag_skip_identity = True ):
        """
        Analogue of bfs_growth

        :param moves              : Tensor of possible moves
        :param start_states       : Tensor of points we start from or None, if we collecting many moves in one
        :param radius             : int - how many steps to make
        :param initial_distance   : number - distance to start from
        :param flag_skip_identity : boolean / if True - drop identity move, if False - distance is set to 0

        :return: states and hashes and distances
        """

        if start_states is None:
            start_states = moves.detach().clone()
        else:
            assert (len(start_states.shape) == len(moves.shape)) and (start_states.shape[1] == moves.shape[1]), 'moves and start states are not match'

        act_states = start_states.detach().clone()

        all_hashes = self.make_hashes( start_states )
        all_dists  = initial_distance*torch.ones_like(all_hashes, device=self.device, dtype=self.dtype_for_dist)
        all_states = start_states

        for r in range(radius):
            act_states = get_neighbors2( act_states.to(self.dtype), moves.to(self.dtype_generators) )#.flatten(end_dim=1)

            act_states, act_hashes, _ = self.get_unique_states_2(act_states)

            mask       = ~torch.isin(act_hashes, all_hashes, assume_unique=True)
            act_states = act_states[mask, :]
            if act_states.shape[0] == 0:
                break

            all_states = torch.vstack([all_states, act_states                                                                            ])
            all_hashes = torch.hstack([all_hashes,                                            act_hashes[mask]                           ])
            all_dists  = torch.hstack([all_dists , (initial_distance + r + 1)*torch.ones_like(act_hashes[mask], dtype=self.dtype_for_dist)])

        id_hash = self.make_hashes( self.state_destination )

        if flag_skip_identity:
            all_states = all_states[all_hashes!=id_hash, :]
            all_dists  = all_dists [all_hashes!=id_hash   ]
            all_hashes = all_hashes[all_hashes!=id_hash   ]
        else:
            all_dists[all_hashes==id_hash] = 0

        return all_states, all_hashes, all_dists
    
    ################################################################################################################################################################################################################################################################################
    def explode_moves_no_states( self, moves, start_states = None, radius=1, initial_distance=1, flag_skip_identity = True ):
        """
        Analogue of bfs_growth

        :param moves              : Tensor of possible moves
        :param start_states       : Tensor of points we start from or None, if we collecting many moves in one
        :param radius             : int - how many steps to make
        :param initial_distance   : number - distance to start from
        :param flag_skip_identity : boolean / if True - drop identity move, if False - distance is set to 0

        :return: states and hashes and distances
        """

        if start_states is None:
            start_states = moves.detach().clone()
        else:
            assert (len(start_states.shape) == len(moves.shape)) and (start_states.shape[1] == moves.shape[1]), 'moves and start states are not match'

        act_states = start_states.detach().clone()

        all_hashes = self.make_hashes( start_states )
        all_dists  = initial_distance*torch.ones_like(all_hashes, device=self.device, dtype=self.dtype_for_dist)
        all_states = start_states

        for r in range(radius):
            act_states, act_hashes, _ = self.get_unique_states_2(get_neighbors2( act_states.to(self.dtype), moves.to(self.dtype_generators) ))#.flatten(end_dim=1)
            mask       = ~torch.isin(act_hashes, all_hashes, assume_unique=True)
            act_states = act_states[mask, :]
            if act_states.shape[0] == 0:
                break

            all_hashes = torch.hstack([all_hashes,                                            act_hashes[mask]                           ])
            all_dists  = torch.hstack([all_dists , (initial_distance + r + 1)*torch.ones_like(act_hashes[mask], dtype=self.dtype_for_dist)])

        id_hash = self.make_hashes( self.state_destination )

        if flag_skip_identity:
            all_dists  = all_dists [all_hashes!=id_hash   ]
            all_hashes = all_hashes[all_hashes!=id_hash   ]
        else:
            all_dists[all_hashes==id_hash] = 0

        return all_hashes, all_dists
    
    ################################################################################################################################################################################################################################################################################
    def register_predictor(self, models_or_heuristics, batching = False, lazy = True, verbose = 0):
        """
        Add model to permutation group for speed up of runs of beam search
        
        :param models_or_heuristics : nn.Module or Predictor or str or function or other model
        :param batching             : boolean / use batching or not - batching suits only 3M/4M models, need to rewrite for other purposes
        :param lazy                 : do not reload if models_or_heuristics is the same that already loaded
        """
        if (self.predictor is not None) and hasattr(self.predictor, 'models_or_heuristics') and (self.predictor.models_or_heuristics == models_or_heuristics) and lazy:
            #if verbose >= 0:
            #    print(f'{models_or_heuristics} already registered, if you want to reload it - set lazy to False')
            pass
        elif isinstance(models_or_heuristics, Predictor):
            if models_or_heuristics.device == self.device:
                self.predictor = models_or_heuristics
            else:
                raise ValueError('Wrong predictor device will be slowing down processing')
        else:
            self.predictor = Predictor( models_or_heuristics, batching = batching, device = self.device )

    ################################################################################################################################################################################################################################################################################
    def beam_search_permutations_torch( self                                    ,
                                       
                                        state_start                             ,
                                        
                                        # model, if not registered
                                        models_or_heuristics            = None  ,
                                        models_or_heuristics_batching   = True  ,            # better not to change
                                        models_or_heuristics_lazy       = True  ,            # better not to change
                                        models_or_heuristics_need_state_destination = False, # better not to change
                                                                                             # models_or_heuristics_bla_bla_bla vars are not a best solution and mb will be changed in future 
                                        # main params
                                        beam_width                      = 1_000 ,
                                        n_steps_limit                   = 1_000 ,
                                        
                                        # if do many basic moves in one step params
                                        n_step_size                     = 1     , 
                                        n_beam_candidate_states         = 'Auto',
                                       
                                        # bi-bfs params
                                        radius_destination_neigbourhood = 5     ,
                                        radius_beam_neigbourhood        = 0     ,
                                        bi_bfs_chunk_size_states        = 1000  ,
                                        mode_bibfs_checks               = (10,5), # experimental - start rev-bi-bfs checks from step mode_bibfs_checks[0] each mode_bibfs_checks[1] step

                                        # point estimation params
                                        temperature                     = 0.01   ,
                                        temperature_decay               = 0.95  ,
                                        alpha_past_states_attenuation   = 0.2   ,
                                        
                                        # backtracking params           
                                        n_steps_to_ban_backtracking     = 8     ,
                                        flag_empty_backtracking_list    = False ,
                                        flag_ban_all_seen_states        = False ,
                                       
                                        # retry params
                                        n_attempts_limit                = 3     ,
                                        do_check_stagnation             = True  ,
                                        stagnation_steps                = 8     ,
                                        stagnation_thres                = 0.05  ,
                                       
                                        # sprouting params
                                        n_random_start_steps            = 5     ,

                                        # return path
                                        flag_path_return                = False ,

                                        # diversity
                                        diversity_func                  = hamming_dist,
                                        diversity_weight                = 0.001 ,
                                       
                                        # split_trajectories
                                        sub_ray_split                   = False ,
                                        flag_return_line_stat           = False ,
                                       
                                        # Technical: 
                                        random_seed                     = 'Auto',
                                        verbose                         = 0     ,
                                        print_min_each_step_no          = 0     ,
                                         
                                        free_memory_func                = lambda: None, # can be free_memory
                                       
                                        print_mode                      = None, # 'Beam stat each step'
                                      ):
        '''
        Find path from the "state_start" to the "state_destination" via beam search.
        
        :param state_start                     : torch.tensor - from where to search
                                        
        # model, if not registered
        :param models_or_heuristics            : nn.Module / Predictor / str / function / something with .predict or .__call__ methods
        :param models_or_heuristics_batching   : boolean / do we need batching of data for a model
        :param models_or_heuristics_lazy       : boolean / do we must reload model

        # main params
        :param beam_width                      : int / ray size
        :param n_steps_limit                   : int / max steps in one retry

        # if do many basic moves in one step params
        :param n_step_size                     : int / how many basic moves in one algorithm step
        :param n_beam_candidate_states         : 'Auto' or int / N basic moves in one step - it's about len(moves)**N of total possible steps, so we take just a subset of them, 'Auto' = len(moves)

        # bi-bfs params
        :param radius_destination_neigbourhood : int / raduius of neighbours of destination to precompute and check intersection each step

        # point estimation params
        :param temperature                     : number / tau of gumbel_softmax - setting to near zero makes all directons equally possible
        :param alpha_past_states_attenuation   : number / weight of momentum, 0 - for pure Markov process

        # backtracking params           
        :param n_steps_to_ban_backtracking     : int / number of previous steps to prohibit intersection with
        :param flag_empty_backtracking_list    : bool / if retried - should we empty list of prohibited points

        # retry params
        :param n_attempts_limit                : int / how many tries to do
        :param do_check_stagnation             : bool / True if check stagnation
        
        :param stagnation_steps                : int / if minimum of predictor variates not that much in pairs of steps - do retry
        :param stagnation_thres                : number / if minimum of predictor variates not that much in pairs of steps - do retry

        # sprouting params
        :param n_random_start_steps            : int / how many steps to sprout

        # Technical: 
        :param random_seed                     : int or 'Auto' or 'Skip' / Skip to do nothing, Auto to change to random seed and print it, int - if we print it
        :param verbose                         : int / contols how many text output during the exection / I personally don't use it, but can be useful
                                                 0 - no output
                                                 1  - dots on each step and short info on finish
                                                 10  - total timing for each step
                                                 1000 - detailed time profiling information
                                                 10000 - advanced time profiling - including get_unique_states_2
        '''
#         try:
        # determine random seed
        if   random_seed == 'Auto':
            new_random_seed = setup_of_random(verbose=verbose)
        elif random_seed == 'Skip':
            pass
        else:
            new_random_seed = setup_of_random(random_seed, verbose=verbose)

        # basic inits
        flag_found_destination = False
        res_distance           = None

        if temperature is None:
            temperature = 0.
        temperature_reserve = temperature

        distance_minimums      = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
        estimate_min_on_ray    = torch.tensor( 1000000.  , dtype=torch.float, device=self.device)

        # start point of search  
        if isinstance( state_start, torch.Tensor ):
            state_start =  state_start.to(self.device).to(self.dtype).reshape(-1,self.state_size)
        else:
            state_start = torch.tensor( state_start, device=self.device, dtype = self.dtype).reshape(-1,self.state_size)

        # how many of moves use in one step
        if n_beam_candidate_states == 'Auto':
            n_beam_candidate_states = len(self.list_generators)
        elif not isinstance(n_beam_candidate_states,int):
            raise ValueError(f'Wrong {n_beam_candidate_states}')

        if models_or_heuristics is None and self.predictor is None:
            raise ValueError(f'models_or_heuristics must be set for search')
        elif models_or_heuristics is not None:
            self.register_predictor(  models_or_heuristics         ,
                                      models_or_heuristics_batching,
                                      models_or_heuristics_lazy    ,
                                      verbose = verbose            )
        ##########################################################################################
        # calc actual moves if we're making more than one basic step in one step 
        ##########################################################################################

        beam_moves, _, beam_dists = self.explode_moves(  self.tensor_generators          ,
                                                         initial_distance   = 1          ,
                                                         radius             = n_step_size,
                                                         flag_skip_identity = True       )
        beam_moves = beam_moves[beam_dists==n_step_size]
        beam_dists = beam_dists[beam_dists==n_step_size]
        # we can make steps of different length, but calculating a final path length can be a bit messy right now

        ##########################################################################################
        # r1-area of state_destination
        ##########################################################################################

        if isinstance(radius_destination_neigbourhood, int): 
            dest_ahash, dest_dists = self.explode_moves_no_states( self.tensor_generators,
                                                                   start_states       = self.state_destination.reshape((-1,self.state_size)),
                                                                   initial_distance   = 0                                                   ,
                                                                   radius             = radius_destination_neigbourhood                     ,
                                                                   flag_skip_identity = False                                               )
    
            dest_ahash, idx = dest_ahash.sort(stable=True)
            dest_dists      = dest_dists[idx]
        elif isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
            dest_states, dest_dists  = self.random_walks( radius_destination_neigbourhood[0], radius_destination_neigbourhood[1] )
            _, dest_ahash, idx       = self.get_unique_states_2(dest_states)
            dest_dists               = dest_dists[idx]

        ##########################################################################################
        # r2-moves from ray front
        ##########################################################################################
        if (radius_beam_neigbourhood > 0):
            rbfs_moves, _, rbfs_dists = self.explode_moves( self.tensor_generators                   ,
                                                            initial_distance   = 1                       ,
                                                            radius             = radius_beam_neigbourhood,
                                                            flag_skip_identity = True                    )
            rbfs_moves = rbfs_moves[rbfs_dists==radius_beam_neigbourhood] # looks like I've forgot this at the first time
            rbfs_dists = rbfs_dists[rbfs_dists==radius_beam_neigbourhood] # looks like I've forgot this at the first time

        ##########################################################################################
        # preparation for stagnation check
        ##########################################################################################
        if do_check_stagnation:
            hashes_stagnation       = torch.zeros(4, beam_width, dtype=self.dtype_for_hash, device=self.device)
            states_bad_hashed       = torch.zeros(1, beam_width, dtype=self.dtype_for_hash, device=self.device)

        if   n_steps_to_ban_backtracking > 0 and not flag_ban_all_seen_states:
            hashes_previous_n_steps = torch.zeros((n_steps_to_ban_backtracking                    , beam_width), dtype=self.dtype_for_hash, device=self.device)
        elif n_steps_to_ban_backtracking > 0 and     flag_ban_all_seen_states:
            hashes_previous_n_steps = torch.zeros((n_steps_to_ban_backtracking * self.n_generators, beam_width), dtype=self.dtype_for_hash, device=self.device)

        ##########################################################################################
        # Loop over attempts (restarts)
        ##########################################################################################
        for i_attempt in range(0, n_attempts_limit):
            if flag_found_destination:
                break
                
            if verbose>0:
                print('>')

            #print(f"({i_attempt})")

            temperature = temperature_reserve/temperature_decay

            prev_states_eval = None

            reach_hashes     = None
            reach_dists      = None

            if flag_path_return:
                if radius_beam_neigbourhood > 0:
                    raise ValueError(f'radius_beam_neigbourhood is {radius_beam_neigbourhood} > 0 - not supported with path return right now')
                if isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
                    raise ValueError(f'radius_destination_neigbourhood is random-walks-like with params {radius_destination_neigbourhood} - not supported with path return right now')
                if n_step_size != 1:
                    raise ValueError(f'n_step_size is {n_step_size} != 1 - not supported with path return right now')
                dict_path = {'radius_destination_neigbourhood': radius_destination_neigbourhood,
                             'n_random_start_steps'           : n_random_start_steps           ,
                             'state_start'                    : state_start                    ,
                             #'radius_beam_neigbourhood'       : radius_beam_neigbourhood       ,
                            }

            # Initialize array of states 
            array_of_states = state_start.view( -1, self.state_size  ).clone().to(self.dtype).to(self.device)

            # Get stochastic state to start from
            for i_tmp in range(n_random_start_steps):
                array_of_states          = get_neighbors2(array_of_states, self.tensor_generators )#.flatten(end_dim=1)
                array_of_states, hshs, _ = self.get_unique_states_2(array_of_states)
                _perm                    = torch.randperm(array_of_states.shape[0], device=self.device)[:min(beam_width-1,array_of_states.shape[0])]
                array_of_states          = array_of_states[_perm, :]

                if flag_path_return:
                    dict_path[-1*i_tmp-1] = hshs[_perm][:min(beam_width-1,array_of_states.shape[0])].detach().cpu()

            if n_random_start_steps>0:
                array_of_states       = torch.vstack([state_start.view( -1, self.state_size  ), array_of_states,]).to(self.device)

            array_of_states_dists     = torch.ones_like(array_of_states[:,0], device=self.device, dtype=torch.int16)*n_random_start_steps
            array_of_states_dists[0]  = 0

            # Initialize hashed history
            hashed_start = self.make_hashes( array_of_states )
            
            hashes_to_sep_lines = torch.arange(len(hashed_start), device=self.device, dtype=torch.int32)

            if flag_path_return:
                dict_path[0] = hashed_start.detach().cpu()

            if n_steps_to_ban_backtracking > 0:
                if flag_empty_backtracking_list:
                    if   not flag_ban_all_seen_states:
                        hashes_previous_n_steps = torch.zeros((n_steps_to_ban_backtracking                    , beam_width), dtype=self.dtype_for_hash, device=self.device)
                    elif     flag_ban_all_seen_states:
                        hashes_previous_n_steps = torch.zeros((n_steps_to_ban_backtracking * self.n_generators, beam_width), dtype=self.dtype_for_hash, device=self.device)

                hashes_previous_n_steps[0,:len(hashed_start)] = hashed_start

            if (self.device == torch.device("cuda")) : torch.cuda.synchronize()

            free_memory_func()    

            ##########################################################################################
            # Main Loop over steps
            ##########################################################################################
            #print(n_steps_limit)
            for i_step in range(1,n_steps_limit+1):
                if (verbose >= 1 ): print('.', end='')

                t_moves = t_estimate =t_check= t_hash = t_isin = t_unique_els = 0; t_all = time.time() # Time profiling

                ### IT'S TESTED AND NOT GIVING ANY PROGRESS WHILE WASTING TIME 
                #if isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
                #    dest_states, dest_dists  = self.random_walks( radius_destination_neigbourhood[0], radius_destination_neigbourhood[1] )
                #    _, dest_ahash, idx       = self.get_unique_states_2(dest_states)
                #    dest_dists               = dest_dists[idx]

                temperature = temperature*temperature_decay

                #choose  n_beam_candidate_states moves
                gen_indx = torch.randperm(len(beam_moves), device=self.device)[:n_beam_candidate_states]

                # Apply generator to all current states 
                t1 = time.time()

                free_memory_func()

                array_of_states_new   = get_neighbors2(array_of_states, beam_moves[gen_indx,:] )#.flatten(end_dim=1)
                prev_states_eval      = prev_states_eval     .repeat(n_beam_candidate_states,1).T.reshape((-1,1)).flatten() if prev_states_eval is not None else torch.zeros_like(array_of_states_new[:,0], device=self.device, dtype=torch.float)
                array_of_states_dists = array_of_states_dists.repeat(n_beam_candidate_states,1).T.reshape((-1,1)).flatten() + n_step_size
                hashes_to_sep_lines   = hashes_to_sep_lines  .repeat(n_beam_candidate_states,1).T.reshape((-1,1)).flatten()

                free_memory_func()

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_moves += (time.time() - t1)        

                # Take only unique states 
                # surprise: THAT IS CRITICAL for beam search performance !!!!
                # if that is not done - beam search  will not find the desired state - quite often
                # The reason - essentianlly beam can degrade, i.e. can be populated by copy of only one state 
                # It is surprising that such degradation  happens quite often even for beam_width = 10_000 - but it is indeed so  
                t1 = time.time()

                array_of_states_new, hashes_array_of_states_new, _idxs = self.get_unique_states_2(array_of_states_new)
                prev_states_eval      = prev_states_eval     [_idxs]
                array_of_states_dists = array_of_states_dists[_idxs]
                hashes_to_sep_lines   = hashes_to_sep_lines  [_idxs]

                free_memory_func()

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_unique_els += (time.time()-t1)
                # filter states from last n_steps_to_ban_backtracking restricted to visit
                mask = torch.ones(hashes_array_of_states_new.shape[0], dtype=torch.bool, device=self.device)

                if n_steps_to_ban_backtracking > 0:
                    mask                      &= ~torch.isin(hashes_array_of_states_new, hashes_previous_n_steps[hashes_previous_n_steps!=0], assume_unique=True)

                if do_check_stagnation and (states_bad_hashed!=0).sum()>0:
                    mask                      &= ~torch.isin(hashes_array_of_states_new, states_bad_hashed[states_bad_hashed!=0], assume_unique=True)

                if n_steps_to_ban_backtracking > 0 and     flag_ban_all_seen_states:
                    hashes_previous_n_steps[i_step    % n_steps_to_ban_backtracking, :len(hashes_array_of_states_new)] = hashes_array_of_states_new

                _ms = mask.sum()
                if _ms<mask.shape[0]:
                    array_of_states_new        = array_of_states_new       [mask,:]
                    prev_states_eval           = prev_states_eval          [mask  ]
                    hashes_array_of_states_new = hashes_array_of_states_new[mask  ]
                    array_of_states_dists      = array_of_states_dists     [mask  ]
                    hashes_to_sep_lines        = hashes_to_sep_lines       [mask  ]
                elif _ms == 0:
                    if verbose>=1:
                        print('!', end='')
                    break

                free_memory_func()    

                # Check destination state found 
                t1 = time.time()
                
                if (radius_beam_neigbourhood > 0) and (i_step >= mode_bibfs_checks[0]) and (i_step%mode_bibfs_checks[1] == 0):
                    reach_hashes, reach_dists = self.search_neighbors_for_destination_reach(array_of_states_new                                ,
                                                                                            rbfs_moves                                         ,
                                                                                            stopping_criteria_hashed = dest_ahash              ,
                                                                                            states_distances         = array_of_states_dists   ,
                                                                                            moves_distances          = rbfs_dists              ,
                                                                                           )
                    if len(reach_hashes)>0:
                        flag_found_destination = True
                else:
                    # Check destination state found 
                    mask = isin_via_searchsorted(hashes_array_of_states_new.flatten(), dest_ahash)
                    if mask.sum()>0:
                        reach_hashes = hashes_array_of_states_new[mask]
                        reach_dists  = array_of_states_dists     [mask]
                        flag_found_destination = True

                    
                free_memory_func()

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_check += (time.time() - t1)
                t1 = time.time()

                if flag_found_destination:
                    # next 4 rows are in need for calc minimal found distance
                    dest_ahash_m = torch.isin(  dest_ahash  , reach_hashes, assume_unique=True ) # isin_via_searchsorted(dest_ahash, reach_hashes)

                    if flag_path_return:
                        dict_path[i_step] = reach_hashes.detach().cpu()
                        dict_path['ended_at_step'] = i_step

                    dest_ahash   = torch.vstack([dest_ahash[dest_ahash_m], torch.zeros_like(dest_ahash[dest_ahash_m]), dest_dists [dest_ahash_m]])
                    reach_hashes = torch.vstack([reach_hashes            , torch.ones_like( reach_hashes)            , reach_dists              ])

                    reach_df     = torch.hstack([dest_ahash, reach_hashes]).T # if reach_hashes isn't deduplicated - this step will break
                    reach_df     = reach_df[reach_df[:,0].argsort(stable=True), :]

                    # print(reach_df)

                    # ok, this is a final distance
                    res_distance = (reach_df[0::2, 2]+reach_df[1::2, 2]).min() #i_step*n_step_size
                    if (verbose >= 10 ):
                        print('Found destination state. ', 'i_step:', i_step, ' distance:', res_distance) 
                    break

                free_memory_func()

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_check += (time.time() - t1)

                # Estimate distance of new states to the destination state (or best moves probabilities for policy models)
                t1 = time.time()

                estimations_for_new_states = self.predictor(array_of_states_new, self)
                #print_estimations_for_new_states
                if print_mode == 'Beam stat each step':
                    if (i_step <100) or ( ((i_step % 10) == 0) and (i_step < 1000) )  or ( ((i_step % 100) == 0)  ): 
                        print('i_step:',i_step, 'Beam Min: %.6f'%torch.min(estimations_for_new_states).item(),'Max: %.6f'%torch.max(estimations_for_new_states).item(), 
                              'Mean: %.6f'%torch.mean(estimations_for_new_states).item(),'Median: %.6f'%torch.median(estimations_for_new_states).item())
                #print(    'All:', estimations_for_new_states)
                estimate_min_on_ray        = estimations_for_new_states.min()#torch.minimum(estimate_min_on_ray, estimations_for_new_states.min())
                
                estimations_for_new_states = estimations_for_new_states - estimate_min_on_ray + 0.001

                # diversity added V1
                #if diversity_weight is not None:
                #    diversity_addon        = diversity_func( array_of_states_new, array_of_states_new[estimations_for_new_states.argmin(),:].view((1,-1)) ) * diversity_weight
                
                # Take only "beam_width" of the best states (i.e. most nearest to destination according to the model estimate)
                ### UPDATED - use softmax w temperature and naive sampling

                prev_states_eval          *= alpha_past_states_attenuation
                prev_states_eval          += (1.-alpha_past_states_attenuation)*(estimations_for_new_states).log()

                estimations_for_new_states  = prev_states_eval+torch.randn_like(prev_states_eval)*(prev_states_eval.max() - prev_states_eval.min())*temperature/2

                # diversity added V2
                if diversity_weight is not None:
                    estimations_for_new_states += diversity_func( array_of_states_new, array_of_states_new[estimations_for_new_states.argmin(),:].view((1,-1)) ) * diversity_weight

                idx0 = torch.argsort( array_of_states_dists, stable=True)

                # split_trajectories
                zeros_to_sep_lines      = torch.zeros_like(hashes_to_sep_lines, dtype=torch.float32, device=self.device)
                if sub_ray_split:
                    idx1 = torch.argsort(estimations_for_new_states[idx0], stable=True)
                    _, _, idx2               = self.get_unique_states_2(hashes_to_sep_lines[idx0][idx1], True)
                    zeros_to_sep_lines      += 1_000_000.
                    zeros_to_sep_lines[idx2] = 0.
                else:
                    idx1 = torch.arange(len(estimations_for_new_states))

                idx = torch.argsort(estimations_for_new_states[idx0][idx1]+zeros_to_sep_lines, stable=True)[:beam_width]
                
                array_of_states       = array_of_states_new       [idx0,:][idx1,:][idx,:]
                hashes_previous       = hashes_array_of_states_new[idx0  ][idx1  ][idx  ]
                prev_states_eval      = prev_states_eval          [idx0  ][idx1  ][idx  ]
                array_of_states_dists = array_of_states_dists     [idx0  ][idx1  ][idx  ]
                hashes_to_sep_lines   = hashes_to_sep_lines       [idx0  ][idx1  ][idx  ]

                if flag_path_return:
                    dict_path[i_step] = hashes_previous.detach().cpu()
                
                distance_minimums = torch.hstack([distance_minimums, prev_states_eval.min(),])
                
                if print_min_each_step_no>0 and i_step%print_min_each_step_no == 0:
                    print(f'Min on ray: {estimate_min_on_ray}')
                    #if sub_ray_split:
                    #    print(hashes_to_sep_lines.min(), hashes_to_sep_lines.max())

                free_memory_func()

                if n_steps_to_ban_backtracking > 0 and not flag_ban_all_seen_states:
                    #hashes_previous_n_steps[i_step    % n_steps_to_ban_backtracking, :] = 0 # turned off but saved for simplicity
                    hashes_previous_n_steps[i_step    % n_steps_to_ban_backtracking, :len(hashes_previous)] = hashes_previous

                if do_check_stagnation:
                    hashes_stagnation[     (i_step-1) % 4                          , :                    ] = 0
                    hashes_stagnation[     (i_step-1) % 4                          , :len(hashes_previous)] = hashes_previous

                    if (i_step >= 4) and (i_step % 4 == 0) and (i_attempt < n_attempts_limit-1) and (check_stagnation(hashes_stagnation, i_step)):
                        states_bad_hashed = torch.vstack([states_bad_hashed,
                                                          hashes_stagnation[(i_step-1) % 4, :].reshape((1,-1)),
                                                          hashes_stagnation[(i_step-2) % 4, :].reshape((1,-1)),
                                                         ])
                        distance_minimums = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
                        if (states_bad_hashed[0,:] == 0).all():
                            states_bad_hashed = states_bad_hashed[1:,:]

                        if verbose>=1:
                            print('!', end='')
                        break

                    elif (i_step >= stagnation_steps) and\
                         (distance_minimums[-stagnation_steps:  ].max()-distance_minimums[-stagnation_steps:  ].min() < stagnation_thres) and\
                         (distance_minimums[-stagnation_steps::2].max()-distance_minimums[-stagnation_steps::2].min() < stagnation_thres):
                        states_bad_hashed = torch.vstack([states_bad_hashed,
                                                          hashes_stagnation[(i_step-1) % 4, :].reshape((1,-1)),
                                                          hashes_stagnation[(i_step-2) % 4, :].reshape((1,-1)),
                                                         ])
                        distance_minimums = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
                        if (states_bad_hashed[0,:] == 0).all():
                            states_bad_hashed = states_bad_hashed[1:,:]

                        if verbose>=1:
                            print('!', end='')
                        break

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_estimate += (time.time() - t1)
                t_all = (time.time() - t_all )
                if verbose >= 1000:
                    print(i_step,'i_step', 't_moves: %.5f, '%t_moves,  't_check: %.3f, '%t_check, 't_unique_els: %.3f, '%t_unique_els, 
                          't_estimate: %.5f, '%t_estimate,  't_all: %.3f, '%t_all, 
                          array_of_states_new.shape, 'array_of_states_new.shape' )
                elif verbose >= 10:
                    print(i_step,'i_step', 't_all: %.3f, '%t_all, array_of_states_new.shape, 'array_of_states_new.shape' )

        res_distance = res_distance.item() if res_distance is not None and flag_found_destination else i_step*n_step_size+(radius_destination_neigbourhood if isinstance(radius_destination_neigbourhood, int) else radius_destination_neigbourhood[0])+radius_beam_neigbourhood

        dict_additional_data = {'random_seed':new_random_seed,}        
        if verbose >= 1:
            print();
            print('Search finished.', 'beam_width:', beam_width)
            if flag_found_destination:    
                print(res_distance, ' steps to destination state. Path found.')
            else:
                print('Path not found.')

        if flag_found_destination and flag_path_return:
            result_states, result_hashes = self.path_cleanup( dict_path )
            dict_additional_data['path_states'] = result_states
            dict_additional_data['path_hashes'] = result_hashes
            #dict_additional_data['path'] = dict_path
            
        if flag_return_line_stat:
            dict_additional_data['line_idxs'] = hashes_to_sep_lines
            
        return flag_found_destination, res_distance, dict_additional_data
#         except Exception as e:
#             print(e)
#             for obj in gc.get_objects():
#                 try:
#                     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                         print( obj.size(), obj.dtype, obj.device, obj.element_size() * obj.nelement() )
#                 except:
#                     pass
#         finally:
#             free_memory()

    
    ################################################################################################################################################################################################################################################################################
    def beam_search_permutations_torch_ME( self                                    ,
                                       
                                        state_start                             ,
                                        
                                        # model, if not registered
                                        models_or_heuristics            = None  ,
                                        models_or_heuristics_batching   = True  ,            # better not to change
                                        models_or_heuristics_lazy       = True  ,            # better not to change
                                        models_or_heuristics_need_state_destination = False, # better not to change
                                                                                             # models_or_heuristics_bla_bla_bla vars are not a best solution and mb will be changed in future 
                                        # main params
                                        beam_width                      = 1_000 ,
                                        n_steps_limit                   = 1_000 ,
                                        
                                        # if do many basic moves in one step params
                                        n_step_size                     = 1     , 
                                        n_beam_candidate_states         = 'Auto',
                                       
                                        # bi-bfs params
                                        radius_destination_neigbourhood = 5     ,
                                        radius_beam_neigbourhood        = 0     ,
                                        bi_bfs_chunk_size_states        = 1000  ,
                                        mode_bibfs_checks               = (10,5), # experimental - start rev-bi-bfs checks from step mode_bibfs_checks[0] each mode_bibfs_checks[1] step

                                        # point estimation params
                                        temperature                     = 0.01   ,
                                        temperature_decay               = 0.95  ,
                                        alpha_past_states_attenuation   = 0.2   ,
                                        
                                        # backtracking params           
                                        n_steps_to_ban_backtracking     = 8     ,
                                        flag_empty_backtracking_list    = False ,
                                        #flag_ban_all_seen_states        = False , # must be added later
                                       
                                        # retry params
                                        n_attempts_limit                = 3     ,
                                        do_check_stagnation             = True  ,
                                        stagnation_steps                = 8     ,
                                        stagnation_thres                = 0.05  ,
                                       
                                        # sprouting params
                                        n_random_start_steps            = 5     ,

                                        # return path
                                        flag_path_return                = False ,

                                        # diversity
                                        diversity_func                  = hamming_dist,
                                        diversity_weight                = 0.001 ,
                                       
                                        # Technical: 
                                        random_seed                     = 'Auto',
                                        verbose                         = 0     ,
                                        print_min_each_step_no          = 0     ,
                                         
                                        free_memory_func                = free_memory, #lambda: None, # can be free_memory
                                      ):
        '''
        Find path from the "state_start" to the "state_destination" via beam search.
        
        :param state_start                     : torch.tensor - from where to search
                                        
        # model, if not registered
        :param models_or_heuristics            : nn.Module / Predictor / str / function / something with .predict or .__call__ methods
        :param models_or_heuristics_batching   : boolean / do we need batching of data for a model
        :param models_or_heuristics_lazy       : boolean / do we must reload model

        # main params
        :param beam_width                      : int / ray size
        :param n_steps_limit                   : int / max steps in one retry

        # if do many basic moves in one step params
        :param n_step_size                     : int / how many basic moves in one algorithm step
        :param n_beam_candidate_states         : 'Auto' or int / N basic moves in one step - it's about len(moves)**N of total possible steps, so we take just a subset of them, 'Auto' = len(moves)

        # bi-bfs params
        :param radius_destination_neigbourhood : int / raduius of neighbours of destination to precompute and check intersection each step

        # point estimation params
        :param temperature                     : number / tau of gumbel_softmax - setting to near zero makes all directons equally possible
        :param alpha_past_states_attenuation   : number / weight of momentum, 0 - for pure Markov process

        # backtracking params           
        :param n_steps_to_ban_backtracking     : int / number of previous steps to prohibit intersection with
        :param flag_empty_backtracking_list    : bool / if retried - should we empty list of prohibited points

        # retry params
        :param n_attempts_limit                : int / how many tries to do
        :param do_check_stagnation             : bool / True if check stagnation
        
        :param stagnation_steps                : int / if minimum of predictor variates not that much in pairs of steps - do retry
        :param stagnation_thres                : number / if minimum of predictor variates not that much in pairs of steps - do retry

        # sprouting params
        :param n_random_start_steps            : int / how many steps to sprout

        # Technical: 
        :param random_seed                     : int or 'Auto' or 'Skip' / Skip to do nothing, Auto to change to random seed and print it, int - if we print it
        :param verbose                         : int / contols how many text output during the exection / I personally don't use it, but can be useful
                                                 0 - no output
                                                 1  - dots on each step and short info on finish
                                                 10  - total timing for each step
                                                 1000 - detailed time profiling information
                                                 10000 - advanced time profiling - including get_unique_states_2
        '''
#         try:
        # determine random seed
        # determine random seed
        if   random_seed == 'Auto':
            new_random_seed = setup_of_random(verbose=verbose)
        elif random_seed == 'Skip':
            pass
        else:
            new_random_seed = setup_of_random(random_seed, verbose=verbose)

        # basic inits
        flag_found_destination = False
        res_distance           = None

        if temperature is None:
            temperature = 0.
        temperature_reserve = temperature

        distance_minimums      = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
        estimate_min_on_ray    = torch.tensor( 1000000.  , dtype=torch.float, device=self.device)

        # start point of search  
        if isinstance( state_start, torch.Tensor ):
            state_start =  state_start.to(self.device).to(self.dtype).reshape(-1,self.state_size)
        else:
            state_start = torch.tensor( state_start, device=self.device, dtype = self.dtype).reshape(-1,self.state_size)

        # how many of moves use in one step
        if n_beam_candidate_states == 'Auto':
            n_beam_candidate_states = len(self.list_generators)
        elif not isinstance(n_beam_candidate_states,int):
            raise ValueError(f'Wrong {n_beam_candidate_states}')

        if models_or_heuristics is None and self.predictor is None:
            raise ValueError(f'models_or_heuristics must be set for search')
        elif models_or_heuristics is not None:
            self.register_predictor(  models_or_heuristics         ,
                                      models_or_heuristics_batching,
                                      models_or_heuristics_lazy    ,
                                      verbose = verbose            )
        ##########################################################################################
        # calc actual moves if we're making more than one basic step in one step 
        ##########################################################################################

        beam_moves, _, beam_dists = self.explode_moves(  self.tensor_generators          ,
                                                         initial_distance   = 1          ,
                                                         radius             = n_step_size,
                                                         flag_skip_identity = True       )
        beam_moves = beam_moves[beam_dists==n_step_size]
        beam_dists = beam_dists[beam_dists==n_step_size]
        # we can make steps of different length, but calculating a final path length can be a bit messy right now

        ##########################################################################################
        # r1-area of state_destination
        ##########################################################################################

        if isinstance(radius_destination_neigbourhood, int): 
            dest_ahash, dest_dists = self.explode_moves_no_states( self.tensor_generators,
                                                                   start_states       = self.state_destination.reshape((-1,self.state_size)),
                                                                   initial_distance   = 0                                                   ,
                                                                   radius             = radius_destination_neigbourhood                     ,
                                                                   flag_skip_identity = False                                               )
    
            dest_ahash, idx = dest_ahash.sort(stable=True)
            dest_dists      = dest_dists[idx]
        elif isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
            dest_states, dest_dists  = self.random_walks( radius_destination_neigbourhood[0], radius_destination_neigbourhood[1] )
            _, dest_ahash, idx       = self.get_unique_states_2(dest_states)
            dest_dists               = dest_dists[idx]

        ##########################################################################################
        # r2-moves from ray front
        ##########################################################################################
        if (radius_beam_neigbourhood > 0):
            rbfs_moves, _, rbfs_dists = self.explode_moves( self.tensor_generators                   ,
                                                            initial_distance   = 1                       ,
                                                            radius             = radius_beam_neigbourhood,
                                                            flag_skip_identity = True                    )
            rbfs_moves = rbfs_moves[rbfs_dists==radius_beam_neigbourhood] # looks like I've forgot this at the first time
            rbfs_dists = rbfs_dists[rbfs_dists==radius_beam_neigbourhood] # looks like I've forgot this at the first time

        ##########################################################################################
        # preparation for stagnation check
        ##########################################################################################
        if do_check_stagnation:
            hashes_stagnation       = torch.zeros(4, beam_width, dtype=self.dtype_for_hash, device=self.device)
            states_bad_hashed       = torch.zeros(1, beam_width, dtype=self.dtype_for_hash, device=self.device)

        if n_steps_to_ban_backtracking > 0:
            hashes_previous_n_steps = torch.zeros((n_steps_to_ban_backtracking, beam_width), dtype=self.dtype_for_hash, device=self.device)

        ##########################################################################################
        # Loop over attempts (restarts)
        ##########################################################################################
        for i_attempt in range(0, n_attempts_limit):
            if flag_found_destination:
                break
                
            if verbose>0:
                print('>')

            #print(f"({i_attempt})")

            temperature = temperature_reserve/temperature_decay

            reach_hashes     = None
            reach_dists      = None
            prev_states_eval = None

            if flag_path_return:
                if radius_beam_neigbourhood > 0:
                    raise ValueError(f'radius_beam_neigbourhood is {radius_beam_neigbourhood} > 0 - not supported with path return right now')
                if isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
                    raise ValueError(f'radius_destination_neigbourhood is random-walks-like with params {radius_destination_neigbourhood} - not supported with path return right now')
                if n_step_size != 1:
                    raise ValueError(f'n_step_size is {n_step_size} != 1 - not supported with path return right now')
                dict_path = {'radius_destination_neigbourhood': radius_destination_neigbourhood,
                             'n_random_start_steps'           : n_random_start_steps           ,
                             'state_start'                    : state_start                    ,
                             #'radius_beam_neigbourhood'       : radius_beam_neigbourhood       ,
                            }

            # Initialize array of states 
            array_of_states = state_start.view( -1, self.state_size  ).clone().to(self.dtype).to(self.device)

            # Get stochastic state to start from
            for i_tmp in range(n_random_start_steps):
                array_of_states          = get_neighbors2(array_of_states, self.tensor_generators )#.flatten(end_dim=1)
                array_of_states, hshs, _ = self.get_unique_states_2(array_of_states)
                _perm                    = torch.randperm(array_of_states.shape[0], device=self.device)[:min(beam_width-1,array_of_states.shape[0])]
                array_of_states          = array_of_states[_perm, :]

                if flag_path_return:
                    dict_path[-1*i_tmp-1] = hshs[_perm][:min(beam_width-1,array_of_states.shape[0])].detach().cpu()

            if n_random_start_steps>0:
                array_of_states       = torch.vstack([state_start.view( -1, self.state_size  ), array_of_states,]).to(self.device)

            array_of_states_dists     = torch.ones_like(array_of_states[:,0], device=self.device, dtype=torch.int16)*n_random_start_steps
            array_of_states_dists[0]  = 0

            # Initialize hashed history
            hashed_start = self.make_hashes( array_of_states )

            if flag_path_return:
                dict_path[0] = hashed_start.detach().cpu()

            if n_steps_to_ban_backtracking > 0:
                if flag_empty_backtracking_list:
                    hashes_previous_n_steps = torch.zeros(n_steps_to_ban_backtracking, beam_width).to(self.device)

                hashes_previous_n_steps[0,:len(hashed_start)] = hashed_start

            if (self.device == torch.device("cuda")) : torch.cuda.synchronize()

            free_memory_func()    

            ##########################################################################################
            # Main Loop over steps
            ##########################################################################################
            for i_step in range(1,n_steps_limit+1):
                if (verbose >= 1 ): print('.', end='')

                t_moves = t_estimate =t_check= t_hash = t_isin = t_unique_els = 0; t_all = time.time() # Time profiling

                ### IT'S TESTED AND NOT GIVING ANY PROGRESS WHILE WASTING TIME 
                #if isinstance(radius_destination_neigbourhood, list) or isinstance(radius_destination_neigbourhood, tuple):
                #    dest_states, dest_dists  = self.random_walks( radius_destination_neigbourhood[0], radius_destination_neigbourhood[1] )
                #    _, dest_ahash, idx       = self.get_unique_states_2(dest_states)
                #    dest_dists               = dest_dists[idx]

                temperature = temperature*temperature_decay

                #choose  n_beam_candidate_states moves
                gen_indx = torch.randperm(len(beam_moves), device=self.device)[:n_beam_candidate_states]

                # Apply generator to all current states 
                t1 = time.time()

                free_memory_func()

                array_of_states_cll        = None
                prev_states_eval_cll       = None
                array_of_states_dists_cll  = None
                hashes_array_of_states_cll = None
                estimations_cll            = None

                estimate_min_on_ray        = torch.tensor( 1000000.  , dtype=torch.float, device=self.device)

                ### ITER STARTS HERE
                for j,m in enumerate(gen_indx):
                    array_of_states_new       = get_neighbors2(array_of_states, beam_moves[m:m+1,:] )#.flatten(end_dim=1)
                    #print(array_of_states_new.shape, array_of_states.shape)
                    prev_states_eval_new      = prev_states_eval     .flatten() if prev_states_eval is not None else torch.zeros_like(array_of_states_new[:,0], device=self.device, dtype=torch.float)
                    array_of_states_dists_new = array_of_states_dists.flatten() + n_step_size
    
    
                    array_of_states_new, hashes_array_of_states_new, _idxs = self.get_unique_states_2(array_of_states_new)
                    prev_states_eval_new      = prev_states_eval_new     [_idxs]
                    array_of_states_dists_new = array_of_states_dists_new[_idxs]
                    
                    # filter states from last n_steps_to_ban_backtracking restricted to visit
                    mask = torch.ones(hashes_array_of_states_new.shape[0], dtype=torch.bool, device=self.device)
    
                    if n_steps_to_ban_backtracking > 0:
                        mask                      &= ~torch.isin(hashes_array_of_states_new, hashes_previous_n_steps[hashes_previous_n_steps!=0], assume_unique=True)
    
                    if do_check_stagnation and (states_bad_hashed!=0).sum()>0:
                        mask                      &= ~torch.isin(hashes_array_of_states_new, states_bad_hashed[states_bad_hashed!=0], assume_unique=True)
    
                    _ms = mask.sum()
                    if _ms<mask.shape[0]:
                        array_of_states_new        = array_of_states_new       [mask,:]
                        prev_states_eval_new       = prev_states_eval_new      [mask  ]
                        hashes_array_of_states_new = hashes_array_of_states_new[mask  ]
                        array_of_states_dists_new  = array_of_states_dists_new [mask  ]
                    elif _ms == 0:
                        if verbose>=1:
                            print('!', end='')
                        break
                
                    # calc r2-moves from array_of_states_new and if intersected with dest_ahash - return all intersection points
                    if (radius_beam_neigbourhood > 0) and (i_step >= mode_bibfs_checks[0]) and (i_step%mode_bibfs_checks[1] == 0):
                        reach_hashes_tmp, reach_dists_tmp = self.search_neighbors_for_destination_reach(array_of_states_new                                 ,
                                                                                                rbfs_moves                                          ,
                                                                                                stopping_criteria_hashed = dest_ahash               ,
                                                                                                states_distances         = array_of_states_dists_new,
                                                                                                moves_distances          = rbfs_dists               ,
                                                                                               )
                        if (reach_hashes_tmp is not None) and len(reach_hashes_tmp)>0:
                            flag_found_destination = True
                            reach_hashes = hstack_if_not_None(reach_hashes, reach_hashes_tmp)
                            reach_dists  = hstack_if_not_None(reach_dists,  reach_dists_tmp )
                    else:
                        # Check destination state found 
                        mask = isin_via_searchsorted(hashes_array_of_states_new.flatten(), dest_ahash)
                        if mask.sum()>0:
                            reach_hashes = hstack_if_not_None(reach_hashes, hashes_array_of_states_new[mask])
                            reach_dists  = hstack_if_not_None(reach_dists,  array_of_states_dists_new [mask])
                            flag_found_destination = True

                    if not flag_found_destination:
                        estimations_for_new_states = self.predictor(array_of_states_new, self)
                        #print(estimations_for_new_states)

                        estimate_min_on_chunk = estimations_for_new_states.min()

                        estimate_min_on_ray   = torch.minimum(estimate_min_on_ray, estimate_min_on_chunk)

                        #estimations_for_new_states = (estimations_for_new_states.max()-estimations_for_new_states)#/(estimations_for_new_states.max()-estimations_for_new_states.min())
        
                        # Take only "beam_width" of the best states (i.e. most nearest to destination according to the model estimate)
                        ### UPDATED - use softmax w temperature and naive sampling
        
                        #prev_states_eval_new *= alpha_past_states_attenuation
                        #prev_states_eval_new += (1.-alpha_past_states_attenuation)*(estimations_for_new_states).log()
        
                        #if temperature is not None:
                        #    estimations_for_new_states  = -torch.nn.functional.gumbel_softmax(-prev_states_eval_new.exp(), tau=temperature, hard=False)
                        #else:
                        #    estimations_for_new_states   = prev_states_eval_new.exp()

                        estimations_for_new_states = estimations_for_new_states - estimate_min_on_chunk + 0.001

                        # diversity added V1
                        #if diversity_weight is not None:
                        #    diversity_addon        = diversity_func( array_of_states_new, array_of_states_new[estimations_for_new_states.argmin(),:].view((1,-1)) ) * diversity_weight
        
                        prev_states_eval_new      *= alpha_past_states_attenuation
                        prev_states_eval_new      += (1.-alpha_past_states_attenuation)*(estimations_for_new_states).log()
        
                        estimations_for_new_states  = prev_states_eval_new.detach().clone() + torch.randn_like(prev_states_eval_new)*(prev_states_eval_new.max() - prev_states_eval_new.min())*temperature/2

                        # diversity added V2
                        if diversity_weight is not None:
                            estimations_for_new_states += diversity_func( array_of_states_new, array_of_states_new[estimations_for_new_states.argmin(),:].view((1,-1)) ) * diversity_weight

                        estimations_cll            = hstack_if_not_None(estimations_cll           , estimations_for_new_states)
                        prev_states_eval_cll       = hstack_if_not_None(prev_states_eval_cll      , prev_states_eval_new      )
                    
                    hashes_array_of_states_cll = hstack_if_not_None(hashes_array_of_states_cll, hashes_array_of_states_new)
                    array_of_states_dists_cll  = hstack_if_not_None(array_of_states_dists_cll , array_of_states_dists_new )
                        
                ### ITER ENDS HERE
                free_memory_func()

                ### IF WE FOUND A WAY
                if flag_found_destination:
                    # next 4 rows are in need for calc minimal found distance
                    dest_ahash_m = torch.isin(  dest_ahash  , reach_hashes, assume_unique=True ) # isin_via_searchsorted(dest_ahash, reach_hashes)

                    if flag_path_return:
                        dict_path[i_step] = reach_hashes.detach().cpu()
                        dict_path['ended_at_step'] = i_step

                    dest_ahash   = torch.vstack([dest_ahash[dest_ahash_m], torch.zeros_like(dest_ahash[dest_ahash_m]), dest_dists [dest_ahash_m]])
                    reach_hashes = torch.vstack([reach_hashes            , torch.ones_like( reach_hashes)            , reach_dists              ])

                    reach_df     = torch.hstack([dest_ahash, reach_hashes]).T # if reach_hashes isn't deduplicated - this step will break
                    reach_df     = reach_df[reach_df[:,0].argsort(stable=True), :]

                    # print(reach_df)

                    # ok, this is a final distance
                    res_distance = (reach_df[0::2, 2]+reach_df[1::2, 2]).min() #i_step*n_step_size
                    if (verbose >= 10 ):
                        print('Found destination state. ', 'i_step:', i_step, ' distance:', res_distance) 
                    
                    break

                ### GET ONLY NEEDED STATES AND VALUES FOR NEXT CYCLE
                # ADDITIONAL SORT TO KEEP SHORTED DISTANCES
                idx0 = torch.argsort( array_of_states_dists_cll, stable=True)
                
                idx  = torch.argsort(estimations_cll[idx0], stable=True)[:beam_width]
                hashes_previous = hashes_array_of_states_cll[idx0][idx]
                hashes_previous, idx2 = torch.sort(hashes_previous, stable=True)
                
                for j,m in enumerate(gen_indx):
                    array_of_states_new        = get_neighbors2( array_of_states, beam_moves[m:m+1,:] )
                    hashes_array_of_states_new = self.make_hashes(array_of_states_new)
                    #print(isin_via_searchsorted(hashes_array_of_states_new, hashes_previous))
                    array_of_states_cll        = vstack_if_not_None(array_of_states_cll,
                                                                    array_of_states_new[isin_via_searchsorted(hashes_array_of_states_new, hashes_previous), :])
                    #print('-', array_of_states_cll.shape)
                
                # get_unique_states_2 sort hashes, it must work smoothly
                array_of_states, hashes_previous, idx3 = self.get_unique_states_2(array_of_states_cll)
                
                prev_states_eval                       = prev_states_eval_cll      [idx0][idx][idx2]
                array_of_states_dists                  = array_of_states_dists_cll [idx0][idx][idx2]

                if flag_path_return:
                    dict_path[i_step] = hashes_previous.detach().cpu()

                #print(idx0.shape, idx.shape, idx2.shape, idx3.shape)
                
                ### CONTINUE BASE CODE
                
                distance_minimums = torch.hstack([distance_minimums, prev_states_eval.min(),])
                
                if print_min_each_step_no>0 and i_step%print_min_each_step_no == 0:
                        print(f'Min on ray: {estimate_min_on_ray}')

                if n_steps_to_ban_backtracking > 0:
                    #hashes_previous_n_steps[i_step    % n_steps_to_ban_backtracking, :] = 0 # turned off but saved for simplicity
                    hashes_previous_n_steps[i_step    % n_steps_to_ban_backtracking, :len(hashes_previous)] = hashes_previous

                if do_check_stagnation:
                    hashes_stagnation[     (i_step-1) % 4                          , :                    ] = 0
                    hashes_stagnation[     (i_step-1) % 4                          , :len(hashes_previous)] = hashes_previous

                    if (i_step >= 4) and (i_step % 4 == 0) and (i_attempt < n_attempts_limit-1) and (check_stagnation(hashes_stagnation, i_step)):
                        states_bad_hashed = torch.vstack([states_bad_hashed,
                                                          hashes_stagnation[(i_step-1) % 4, :].reshape((1,-1)),
                                                          hashes_stagnation[(i_step-2) % 4, :].reshape((1,-1)),
                                                         ])
                        distance_minimums = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
                        if (states_bad_hashed[0,:] == 0).all():
                            states_bad_hashed = states_bad_hashed[1:,:]

                        if verbose>=1:
                            print('!', end='')
                        break

                    elif (i_step >= stagnation_steps) and\
                         (distance_minimums[-stagnation_steps:  ].max()-distance_minimums[-stagnation_steps:  ].min() < stagnation_thres) and\
                         (distance_minimums[-stagnation_steps::2].max()-distance_minimums[-stagnation_steps::2].min() < stagnation_thres):
                        states_bad_hashed = torch.vstack([states_bad_hashed,
                                                          hashes_stagnation[(i_step-1) % 4, :].reshape((1,-1)),
                                                          hashes_stagnation[(i_step-2) % 4, :].reshape((1,-1)),
                                                         ])
                        distance_minimums = torch.tensor([1000000.,], dtype=torch.float, device=self.device)
                        if (states_bad_hashed[0,:] == 0).all():
                            states_bad_hashed = states_bad_hashed[1:,:]

                        if verbose>=1:
                            print('!', end='')
                        break

                if (verbose >= 1000 ) and (  self.device == torch.device("cuda")) : torch.cuda.synchronize()
                t_estimate += (time.time() - t1)
                t_all = (time.time() - t_all )
                if verbose >= 1000:
                    print(i_step,'i_step', 't_moves: %.5f, '%t_moves,  't_check: %.3f, '%t_check, 't_unique_els: %.3f, '%t_unique_els, 
                          't_estimate: %.5f, '%t_estimate,  't_all: %.3f, '%t_all, 
                          array_of_states_new.shape, 'array_of_states_new.shape' )
                elif verbose >= 10:
                    print(i_step,'i_step', 't_all: %.3f, '%t_all, array_of_states_new.shape, 'array_of_states_new.shape' )

        res_distance = res_distance.item() if res_distance is not None and flag_found_destination else i_step*n_step_size+(radius_destination_neigbourhood if isinstance(radius_destination_neigbourhood, int) else radius_destination_neigbourhood[0])+radius_beam_neigbourhood

        dict_additional_data = {'random_seed':new_random_seed,}        
        if verbose >= 1:
            print();
            print('Search finished.', 'beam_width:', beam_width)
            if flag_found_destination:    
                print(res_distance, ' steps to destination state. Path found.')
            else:
                print('Path not found.')

        if flag_found_destination and flag_path_return:
            result_states, result_hashes = self.path_cleanup( dict_path )
            dict_additional_data['path_states'] = result_states
            dict_additional_data['path_hashes'] = result_hashes
            #dict_additional_data['path'] = dict_path

        return flag_found_destination, res_distance, dict_additional_data
#         except Exception as e:
#             print(e)
#             for obj in gc.get_objects():
#                 try:
#                     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                         print( obj.size(), obj.dtype, obj.device, obj.element_size() * obj.nelement() )
#                 except:
#                     pass
#         finally:
#             free_memory()
    
    ################################################################################################################################################################################################################################################################################
    def scramble_state(self, n_scrambles ):
        state_current = self.state_destination.detach().clone().reshape((-1))
        for k in range(n_scrambles):
            IX_move = torch.randint(0, self.n_generators, (1,), dtype = self.dtype_generators)#.item() # random moves indixes
            state_current = state_current[ self.list_generators[IX_move] ] 
        return state_current
    
    ################################################################################################################################################################################################################################################################################
    def random_walks(self, n_random_walk_length,  n_random_walks_to_generate):
        '''
        Output:
        returns X,y: X - array of states, y - number of steps rw achieves it

        Input: 
        generators - generators (moves) to make random walks  (permutations), 
            can be list of vectors or array with vstacked vectors
        n_random_walk_length - number of visited nodes, i.e. number of steps + 1 
        n_random_walks_to_generate - how many random walks will run in parrallel
        rw_start - initial states for random walks - by default we will use 0,1,2,3 ...
            Can be vector or array
            If it is vector it will be broadcasted n_random_walks_to_generate times, 
            If it is array n_random_walks_to_generate - input n_random_walks_to_generate will be ignored
                and will be assigned: n_random_walks_to_generate = rw_start.shape[0]

        '''

        array_of_states = self.state_destination.to(self.device).view( 1, self.state_size  ).expand( n_random_walks_to_generate, self.state_size )

        # Output: X,y - states, y - how many steps we achieve them 
        # Allocate memory: 
        X = torch.zeros( n_random_walks_to_generate*n_random_walk_length , self.state_size, device=self.device, dtype = self.dtype )

        # First portion of data  - just our state_rw_start state  multiplexed many times
        X[:n_random_walks_to_generate,:] = array_of_states
        y        = torch.tensor(range(n_random_walks_to_generate*n_random_walk_length), device=self.device)//n_random_walks_to_generate
        IX_moves = torch.randint(0, self.n_generators, (n_random_walks_to_generate*n_random_walk_length-n_random_walks_to_generate,), dtype = torch.int32, device=self.device)

        # Technical to make array[ IX_array] we need  actually to write array[ range(N), IX_array  ]

        # Main loop 
        
        argnr = (torch.arange(n_random_walks_to_generate, dtype=torch.long, device=self.device)*torch.ones((n_random_walks_to_generate,self.tensor_generators.shape[1]), dtype=torch.long, device=self.device).T).T

        for i_step in range(1,n_random_walk_length):
            a,b,c = (i_step-1)*n_random_walks_to_generate, (i_step-0)*n_random_walks_to_generate, (i_step+1)*n_random_walks_to_generate
            #X[ b:c, : ] = X[ a:b, : ][argnr, self.tensor_generators[IX_moves[ a:b ],:]]
            X[ b:c, : ] = torch.gather( X[ a:b, : ], 1, self.tensor_generators[IX_moves[ a:b ],:] )
            
        return X,y
    
    ################################################################################################################################################################################################################################################################################
    def path_cleanup(self, dict_path):
        result_states                   = []
        result_hashes                   = []
    
        radius_destination_neigbourhood = dict_path['radius_destination_neigbourhood']
        n_random_start_steps            = dict_path['n_random_start_steps'           ]
        state_start                     = dict_path['state_start'                    ]
        state_start_hash                = self.make_hashes(state_start)
    
        i_step_max                      = dict_path['ended_at_step']
        from_beam_hashes                = dict_path[i_step_max].to(self.device)
    
        from_dest_states                = self.state_destination.detach().clone()
        from_dest_hashes                = self.make_hashes(from_dest_states)
        from_dest_hashes_store          = []
        from_dest_states_store          = []

        # WE ARE ALWAYS IN SEARCH OF state_start_hash BECAUSE WE ARE MOVING BACKWARDS - FROM state_destination TO state_start
    
        # step 1 - neighbourhood of state_destination - forward
        for j_f in range(0,radius_destination_neigbourhood+1):
            mask = isin_via_searchsorted( from_dest_hashes, state_start_hash )
            if mask.sum() > 0:
                print(f'found {j_f}')
                from_dest_hashes_store.append( state_start_hash )
                from_dest_states_store.append( state_start      )
                return from_dest_states_store, from_dest_hashes_store
            mask = isin_via_searchsorted( from_dest_hashes, from_beam_hashes )
            if mask.sum() == 0:
                from_dest_hashes_store.append( from_dest_hashes )
                from_dest_states_store.append( from_dest_states )
                from_dest_states, from_dest_hashes, _ = self.get_unique_states_2( get_neighbors2(from_dest_states, self.tensor_generators) )
            else:
                from_beam_states      = from_dest_states[mask, :]
                from_beam_hashes      = from_dest_hashes[mask   ]
                from_beam_hashes, idx = from_beam_hashes.sort(stable=True)
                from_beam_states      = from_beam_states[idx , :]
                break
    
        del from_dest_states_store
        
        # step 2 - neighbourhood of state_destination - backward
        result_states.append( from_beam_states.detach().clone() )
        result_hashes.append( from_beam_hashes.detach().clone() )
        
        if j_f > 0:
            # this means that no area of state_destination is precomputed - must go to step 3
            for j_b in range(j_f, 0, -1):
                from_beam_states, from_beam_hashes, _ = self.get_unique_states_2( get_neighbors2(from_beam_states, self.tensor_generators) )
                mask = torch.isin( from_beam_hashes, from_dest_hashes_store[j_b-1] )
                from_beam_states = from_beam_states[mask, :]
                from_beam_hashes = from_beam_hashes[mask   ]
                result_states.append( from_beam_states )
                result_hashes.append( from_beam_hashes )
    
        result_states = list(reversed( result_states ))
        result_hashes = list(reversed( result_hashes ))
    
        # main cycle
        from_beam_states = result_states[-1].detach().clone()
        
        for j_m in range(i_step_max-1, -1, -1):
            mask = isin_via_searchsorted( from_beam_hashes, state_start_hash )
            if mask.sum() > 0:
                print(f'found {j_m}')
                return result_states[:-1]+[state_start,], result_hashes[:-1]+[state_start_hash,]
                
            from_beam_states, from_beam_hashes, _ = self.get_unique_states_2( get_neighbors2(from_beam_states, self.tensor_generators) )
            mask = torch.isin( from_beam_hashes, dict_path[j_m].to(self.device) )
            from_beam_states = from_beam_states[mask, :]
            from_beam_hashes = from_beam_hashes[mask   ]
            result_states.append( from_beam_states )
            result_hashes.append( from_beam_hashes )
    
        # if not found for now - ok, let's search in sprouting steps
        for j_s in range(-n_random_start_steps+1, 0, 1):
            mask = isin_via_searchsorted( from_beam_hashes, state_start_hash )
            if mask.sum() > 0:
                print(f'found {j_s}')
                return result_states[:-1]+[state_start,], result_hashes[:-1]+[state_start_hash,]
                
            from_beam_states, from_beam_hashes, _ = self.get_unique_states_2( get_neighbors2(from_beam_states, self.tensor_generators) )
            mask = torch.isin( from_beam_hashes, dict_path[j_s].to(self.device) )
            from_beam_states = from_beam_states[mask, :]
            from_beam_hashes = from_beam_hashes[mask   ]
            result_states.append( from_beam_states )
            result_hashes.append( from_beam_hashes )
    
        from_beam_states, from_beam_hashes, _ = self.get_unique_states_2( get_neighbors2(from_beam_states, self.tensor_generators) )
    
        mask = isin_via_searchsorted( from_beam_hashes, state_start_hash )
        assert mask.sum() > 0, 'Something went veeeeeeeeerrrrrry wrong((('
        
        # neighbourhood of state_start
        result_states.append( state_start      )
        result_hashes.append( state_start_hash )
        
        return result_states, result_hashes
            
    ################################################################################################################################################################################################################################################################################
    def retractor_base(self, path_states, radius = 3):
        hashes, _ = self.explode_moves_no_states(perm_group_555.tensor_generators, start_states = torch.vstack( path_states ), radius=radius, initial_distance=0, flag_skip_identity=False)
        hashes, _ = hashes.sort(stable=True)
    
        s  = path_states[-1]
        d  = path_states[ 0]
        sh = self.make_hashes( s )
        dh = self.make_hashes( d )
    
        tmp_hashes = [sh,]
    
        shp = None
    
        for i in range(1,len(path_states)):
            mask = isin_via_searchsorted( sh, dh )
            if mask.sum() > 0:
                print(f'found {i-1}')
                break
    
            s, shn, _ = self.get_unique_states_2( get_neighbors2(s, self.tensor_generators) )
            mask      =  isin_via_searchsorted( shn, hashes )
            mask     &= ~isin_via_searchsorted( shn, sh     )
            if shp is not None:
                mask &= ~isin_via_searchsorted( shn, shp    )
            shp = sh.detach().clone()
            s   = s  [mask, :]
            sh  = shn[mask   ]
            tmp_hashes.append( sh )
    
        assert isin_via_searchsorted( sh, dh ).any()
    
        res_states = [d ,]
        res_hashes = [dh,]
        for j in range(len(tmp_hashes)-2,-1,-1):
            d, dh, _ = self.get_unique_states_2( get_neighbors2(d, self.tensor_generators) )
            mask     = isin_via_searchsorted( dh, tmp_hashes[j] )
            d        = d [mask, :]
            res_states.append( d        )
            res_hashes.append( dh[mask] )
    
        return res_states, res_hashes
    
    ################################################################################################################################################################################################################################################################################
    def retractor(self, path_states, radius = 3, max_steps = 1000):
        rs0, rh0 = self.retractor_base(path_states, radius=radius)
        for i in range(max_steps):
            rs1, rh1 = self.retractor_base(rs0, radius=radius)
            if len(rh1) == len(rh0):
                break
            else:
                print(len(rh1), len(rh0))
                rs0 = [q for q in rs1]
                rh0 = [q for q in rh1]
            print(end='@')
        return rs1, rh1
    ################################################################################################################################################################################################################################################################################
    def manhatten_moves_matrix_count(self, steps = 10, bad = 1000, to_power = None, value_dict = None):
        if to_power is not None and value_dict is not None:
            print("value_dict will be ignored because to_power is not None")
        if value_dict is None:
            if to_power is None:
                to_power = 1.6
                
        prms = []
        dsts = []
        
        _prm = get_neighbors2(self.state_destination, self.tensor_generators)
    
        _prm = torch.vstack([self.state_destination                                        , _prm                                                                      ,])
        _dst = torch.hstack([torch.tensor(0, dtype=self.dtype_for_dist, device=self.device), torch.ones_like(_prm[1:,0], dtype=self.dtype_for_dist, device=self.device),])
    
        idx = torch.zeros_like(_dst, dtype=torch.bool)
        
        for j in range(_prm.shape[1]):
            idx[self.get_unique_states_2(_prm[:,j], flag_already_hashed=True)[2]] = True
        
        _prm = _prm[idx, :]
        _dst = _dst[idx   ]
    
        prms.append( _prm[0:1,:] )
        prms.append( _prm[1: ,:] )
    
        dsts.append( _dst[0:1  ] )
        dsts.append( _dst[1:   ] )
    
        for i in range(0, steps):
            #print(prms[-1].shape)
            ### 1
            _prm = get_neighbors2(prms[-1], self.tensor_generators)
    
            _dst = torch.ones_like(_prm[:,0], dtype=self.dtype_for_dist, device=_dst.device)*(i+2)
        
            idx = torch.zeros_like(_dst, dtype=torch.bool)
            
            for j in range(_prm.shape[1]):
                idx[self.get_unique_states_2(_prm[:,j], flag_already_hashed=True)[2]] = True
            
            _prm = _prm[idx, :]
            _dst = _dst[idx   ]
    
            ### 2
            _prm = torch.vstack([torch.vstack(prms), _prm])
            _dst = torch.hstack([torch.hstack(dsts), _dst])
            idx  = torch.zeros_like(_dst, dtype=torch.bool)
            
            for j in range(_prm.shape[1]):
                idx[self.get_unique_states_2(_prm[:,j], flag_already_hashed=True)[2]] = True
            
            _prm = _prm[idx, :][torch.vstack(prms).shape[0]:,:]
            _dst = _dst[idx   ][torch.hstack(dsts).shape[0]:  ]
    
            if _prm.shape[0] == 0:
                break
        
            prms.append( _prm )
            dsts.append( _dst )
    
        #return torch.vstack(prms), torch.hstack(dsts)
    
        _prm = torch.vstack(prms)
        _dst = torch.hstack(dsts)
    
        _vls = torch.ones( (_prm.shape[1],_prm.max()+1), dtype=torch.int64, device=self.device )*bad
        
        for i in range(_prm.shape[0]):
            for j in range(_prm.shape[1]):
                k = _prm[i,j].item()
                if _vls[j,k] == bad:
                    _vls[j,k] = _dst[i].item()

        if to_power is not None:
            self._vls = _vls**to_power
        elif value_dict is not None:
            self._vls = torch.zeros_like(_vls)
            for k,v in value_dict.items():
                self._vls += (_vls == k)*v
        else:
            self._vls = _vls
    
        return _vls
    ################################################################################################################################################################################################################################################################################
    def group_data_0w(self, data):
        return torch.sum(torch.stack([((data==j)*self._vls[:,j]) for j in range(self._vls.shape[1])]), dim=0)
    ################################################################################################################################################################################################################################################################################
    def group_data_1(self, data):
        return torch.sum(torch.stack([((data==j)*self._vls[:,j]) for j in range(self._vls.shape[1])]), dim=0).sum(dim=1)
    ################################################################################################################################################################################################################################################################################
    def pairwise_cmp(self, A, B, p=1, max_moves=None, A_size_thresh = 100):

        #if max_moves is None:
        #    max_moves = self._vls.max().item()
            
        out_v = None
        out_i = None
        for AC in tensor_split_preprocessed(A, A_size_thresh):
            gd0A = self.group_data_0w(AC).to(torch.float)
            gd0B = self.group_data_0w(B ).to(torch.float)
    
            # 3
            res  = (gd0A == 0).to(torch.float) @ (gd0B.T != 0).to(torch.float)
            for i in torch.unique(perm_group_444._vls.flatten()):#range(1,max_moves+1):
                if i > 0:
                    res += (gd0A == i).to(torch.float) @ (gd0B.T != i).to(torch.float)
    
            v, i = res.min(dim=1)
            
            out_v = torch.hstack([out_v, v]) if out_v is not None else v
            out_i = torch.hstack([out_i, i]) if out_i is not None else i
    
        return out_v, out_i
    ################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################
    