"""

FL functionality for learning the FF for the car models


"""

import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy 
from paths import SplinePath
from car_models import BicycleModel
from controller_models import PID, FFmodelData
from utils_control import simulate_closed_loop, simulate_closed_loop_traj_follow

from typing import Dict, Optional
import colormaps as cmaps # type: ignore

def pairwise_cosine_sim(p1, p2):
    """ Calculate the cosine similarity between two parameter vectors. """
    # assert that shape of models is the same
    assert [p1.shape == p2.shape], "Models must have the same shape"
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(p1, p2).item()

def flatten_params(model):
    # Collect all the trainable parameters in a list
    params = [param.view(-1) for param in model.parameters() if param.requires_grad]
    
    # Concatenate them into a single 1D tensor (flattened array)
    flat_params = torch.cat(params, dim=0)
    
    return flat_params

class Node:
    """ handling the local clients for the FL procedure """
    def __init__(self, 
                 id: int, 
                 model: nn.Module, 
                 dynamic_model: str,
                 cfg: dict):
        self.id = id
        self.model = model
        self.cfg = cfg
        self.dynamic_model = dynamic_model

        if dynamic_model == 'bicycle':
            init_state = {'x': 0, 'y': 0, 'psi': 0, 'vl': 0}
            self.dynamics = BicycleModel(initial_state=init_state, cfg=self.cfg)
        elif dynamic_model == 'point':
            pass
        else:
            raise ValueError("Dynamic model must be either 'bicycle' or 'point'.")

    def __str__(self):
        if self.id == -1:
            return f"Server Node"
        else:
            return f"Client {self.id}"
    
    def __repr__(self):
        if self.id == -1:
            return f"Server Node"
        else:
            return f"Client {self.id}"
        
    def _init_car(self, dynamic_model='bicycle', path=None):
        """ Initialize the dynamics model for the given path. """
        if path is None:
            path = self.path 
        init_state = {'x': path.x_fine[0], 'y': path.y_fine[0], 'psi': path.psi_fine[0], 'vl': path.v_long_profile[0]}

        if dynamic_model == 'bicycle':
            self.dynamics = BicycleModel(initial_state=init_state, cfg=self.cfg)

        return self.dynamics
    
    def _init_trajectory(self, path=None):
        if path is None:
            path = self.path
        traj = path.get_trajectory(T_end=self.cfg['simulation']['end_time'], 
                                dt=self.cfg['simulation']['timestep'])
        return traj

class ServerNode(Node):
    def __init__(self, 
                 model: nn.Module,
                 dynamic_model: str,
                 cfg: dict, 
                 id: int = -1): # id of the server is -1):
        super().__init__(id=id, model=model, cfg=cfg, dynamic_model=dynamic_model)
        self.global_models: Dict[str, nn.Module] = {}

        self.rounds = 0
        self.LGC_matrices: dict[str, np.ndarray] = {}

    def fedavg(self, client_messages: list[tuple[nn.Module, dict]]):
        """ Aggregate the models from the local clients. """
        self.rounds += 1
        print(f"Round {self.rounds} of FedAvg for {len(client_messages)} clients ...")
        local_models = [model for model, _ in client_messages]
        nr_data = [info['nr_samples'] for _, info in client_messages]
        assert len(local_models) > 0, "The list of models should not be empty"
        assert len(local_models) == len(nr_data), "The number of models and data points should match"

        # Calculate the weights for the aggregation
        weights = np.array(nr_data) / sum(nr_data)
        
        # Create a copy of the first model to use for the averaged model
        avg_model = copy.deepcopy(local_models[0])
        
        with torch.no_grad():
            for param in avg_model.parameters():
                param.data.zero_()  # Initialize all parameters to zero
                 
        # Sum the parameters from all models
        with torch.no_grad():
            for weight, model in zip(weights, local_models):
                for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                    avg_param.data.add_(weight * param.data)
        
            # # Average the parameters
            # for param in avg_model.parameters():
            #     param.data.div_(len(local_models))
        self.model = avg_model
        self.global_models["round_"+str(self.rounds)] = avg_model
        return 

    def send_global_model(self) -> nn.Module:
        """ Send the global model to the clients. """
        model_copy = self.model._create_copy()
        return model_copy
    
    def eval_global_model(self, path: SplinePath, FF_type='model', FF_input=None) -> tuple[float, pd.DataFrame]:
        """ Evaluate the model with the given data. 
        
        Args:
            path: SplinePath, path for the evaluation (None if the path is the same as the training path)
            FF_type: str, type of FF model {'model', 'analytic', None}
            FF_input: str, input for the FF model {'desired', 'actual'}
        
        Returns:
            mean_traj_error: float, mean deviation from the trajectory
            log_traj: pd.DataFrame, trajectory data
                
        """
        car = self._init_car(dynamic_model=self.dynamic_model,path=path)
        traj = self._init_trajectory(path=path) 
        log_traj = simulate_closed_loop_traj_follow(traj, car, 
                                                       dt=self.cfg['simulation']['timestep'], 
                                                       FF_type=FF_type, FF_model=self.model, FF_input=FF_input)   

        # rmse_dpsi = np.sqrt(np.mean((log_traj['dpsi_actual'] - log_traj['dpsi_desired'])**2))

        mean_traj_error = np.mean(((log_traj['x_d'] - log_traj['x_a'])**2 + (log_traj['y_d'] - log_traj['y_a'])**2)**0.5)

        return mean_traj_error, log_traj
    
    def calc_LCG(self, client_messages):
        """ Calculate the Local Gradient Consistency (LGC) for the given client messages. """
        nr_clients = len(client_messages)

        local_models = [model for model, _ in client_messages]

        if client_messages[0][1]['nr_samples'] == None:
            agg_weights = np.ones(nr_clients, 1)
        else:
            agg_weights = [info['nr_samples'] for _, info in client_messages]
        agg_weights = np.array(agg_weights) / sum(agg_weights)

        # flatten the NN weights for each model
        global_params = flatten_params(self.model)
        pseudo_gradients = torch.zeros((global_params.shape[0], nr_clients))
        for i, local_model in enumerate(local_models):
            p = flatten_params(local_model)
            pseudo_gradients[:,i] = p - global_params.view(-1)
        
        LGC_matrix = np.zeros((nr_clients, nr_clients))
        for i in range(nr_clients):
            for j in range(nr_clients):
                LGC_matrix[i,j] = pairwise_cosine_sim(pseudo_gradients[:,i], pseudo_gradients[:,j])

        self.LGC_matrices['round_'+str(self.rounds)] = LGC_matrix

        LGC = 0
        for i in range(nr_clients):
            for j in range(nr_clients):
                if i != j and j > i:
                    LGC += agg_weights[i] * agg_weights[j] * LGC_matrix[i,j]
        LGC = LGC / nr_clients

        return LGC, LGC_matrix
    
    def plot_LGC_matrix(self, round_number):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.LGC_matrices['round_'+str(round_number)], cmap=cmaps.prinsenvlag.discrete(20), vmin=-1, vmax=1)
        fig.colorbar(cax)
        plt.title(f"LGC matrix for round {round_number}")
        plt.show()
        return


class ClientNode(Node):
    def __init__(self, 
                 id: int, 
                 model: nn.Module, 
                 dynamic_model: str,
                 cfg: dict, 
                 path: SplinePath):
        super().__init__(id=id, model=model, dynamic_model=dynamic_model, cfg=cfg)
        self.optimizer_type = 'Adam'
        self.criterion = nn.MSELoss()
        
        self.train_loader = None
        self.test_loader = None
        
        self.current_data = None
        self.old_data: Dict[str, pd.DataFrame] = {} # store the old data for each round
        self.path = path
        self.epoch_counter = 0

    def simulate_and_learn_FF_local(self, FF_input=None, FF_type='model'):
        """ Simulate the vehicle model with the current FF model and learn the FF model. """
        self.generate_data(FF_input=FF_input, FF_type=FF_type)
        self.local_training(FF_input=FF_input)
        return

    def generate_data(self, FF_type=None, FF_input=None):
        """ Simulate the dynamical system with the current model and the given path. """
        self.epoch_counter += 1
        
        car = self._init_car(dynamic_model=self.dynamic_model, path=self.path)
        traj = self._init_trajectory()
        log_traj_NN = simulate_closed_loop_traj_follow(traj, car, dt=self.cfg['simulation']['timestep'], 
                                                       FF_type=FF_type, 
                                                       FF_model=self.model, 
                                                       FF_input=FF_input)   
        if self.current_data is not None:
            self.old_data["round_"+str(self.epoch_counter-1)] = self.current_data
        self.current_data = log_traj_NN
        return
    
    def local_training(self, FF_input=None):
        """ Train the FF model with the current data. 
        
        Args:
            FF_input: str, input for the FF model {'desired', 'actual'}

        """
        if self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr=self.cfg['NN_model']['learning_rate'])

        # print(f"\tTraining the FF model for client {self.id}")
        dataset = FFmodelData(self.current_data, FF_input=FF_input)
        trainloader = DataLoader(dataset, 
                                 batch_size=self.cfg['NN_model']['batch_size'], 
                                 shuffle=True)
        self.nr_samples = len(trainloader.dataset)
        
        train_losses = []
        self.model.train()
        for epoch in range(self.cfg['learning']['local_epochs']):
            train_loss = []
            for i, (X, y) in enumerate(trainloader):
                optimizer.zero_grad()
                y_pred = self.model(X).flatten()
                loss = self.criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            train_losses.append(np.mean(train_loss))
            print(f"\t\tEpoch: {epoch} | Loss: {train_losses[epoch]**0.5:.6f} | #-samples: {self.nr_samples}") # loss is MSE, print RMSE
        return
    
    def send_model_to_server(self) -> tuple[nn.Module, dict]:
        """ Send the model to the server. """

        add_info = {'nr_samples': self.nr_samples}

        # model = copy.deepcopy(self.model)  # doesn't work when using spectral_norm

        # Manually copy the model's state dictionary
        model_copy = self.model._create_copy()

        return model_copy, add_info
    
    def receive_global_model(self, global_model: nn.Module):
        """ Receive the global model from the server. 
        Overwrite the local model. 
        """
      
        # Iterate over each local model and update its parameters
        with torch.no_grad():  # Disable gradient tracking for the update process
            for glob_param, local_param in zip(global_model.parameters(), self.model.parameters()):
                local_param.data.copy_(glob_param.data)
        return
    
    def eval_model(self, path=None, FF_type=None, FF_input=None) -> tuple[float, pd.DataFrame]:
        """ Evaluate the model with the given data. 
        
        Args:
            path: SplinePath, path for the evaluation (None if the path is the same as the training path)
            FF_type: str, type of FF model {'model', 'analytic', None}
            FF_input: str, input for the FF model {'desired', 'actual'}
        
        Returns:
            mean_traj_error: float, mean deviation from the trajectory
            log_traj: pd.DataFrame, trajectory data
        """
        if path is None:
            path = self.path
        car = self._init_car(dynamic_model=self.dynamic_model, path=path)
        traj = self._init_trajectory(path=path)
        log_traj = simulate_closed_loop_traj_follow(traj, car, 
                                                       dt=self.cfg['simulation']['timestep'],
                                                       FF_type=FF_type, 
                                                       FF_model=self.model, 
                                                       FF_input=FF_input)   

        # rmse_dpsi = np.sqrt(np.mean((log_traj['dpsi_actual'] - log_traj['dpsi_desired'])**2))
        mean_traj_error = np.mean(((log_traj['x_d'] - log_traj['x_a'])**2 + (log_traj['y_d'] - log_traj['y_a'])**2)**0.5)

        return mean_traj_error, log_traj


def eval_FF_model(FF_model, 
                  FF_type, 
                  FF_input,
                  path: SplinePath, 
                  cfg: dict):
    """ Given the current path, evaluate the FF model with the given FF_type.
    Args:
        FF_model: either {torch model} or None
        FF_type: str, type of FF model
        path: SplinePath object 

    Returns:
        mean_traj_error: float, mean deviation from the trajectory
        log_traj: pd.DataFrame, trajectory data 
      
       
    """
    init_state = {'x': path.x_fine[0], 'y': path.y_fine[0], 'psi': path.psi_fine[0], 'vl': path.v_long_profile[0]}
    car = BicycleModel(initial_state=init_state, cfg=cfg)

    traj = path.get_trajectory(cfg['simulation']['end_time'], cfg['simulation']['timestep'])
    log_traj = simulate_closed_loop_traj_follow(traj, car, 
                                                FF_type=FF_type, FF_model=FF_model, FF_input=FF_input)

    # rmse_dpsi = np.sqrt(np.mean((log_traj['dpsi_actual'] - log_traj['dpsi_desired'])**2))
    e_x =  np.cos(log_traj['psi_d']) * (log_traj['x_d'] - log_traj['x_a']) + np.sin(log_traj['psi_d']) * (log_traj['y_d'] - log_traj['y_a'])
    e_y = -np.sin(log_traj['psi_d']) * (log_traj['x_d'] - log_traj['x_a']) + np.cos(log_traj['psi_d']) * (log_traj['y_d'] - log_traj['y_a'])

    mte = np.mean((e_x**2 + e_y**2)**0.5)

    # mean_traj_error = np.mean(((log_traj['x_d'] - log_traj['x_a'])**2 + (log_traj['y_d'] - log_traj['y_a'])**2)**0.5)

    return mte, log_traj


