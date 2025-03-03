"""

Centralized learning with an application of the learned FF model. 

1. Initial simulation with PI-only
2. Learn a FF model from the simulation data
3. Simulate with PI + FF model
4. Learn a FF model from the new simulation data
5. Repeat steps 3-4

Then plot!


"""

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader
import pprint

import os
import json 

from car_models import BicycleModel
from controller_models import FFmodelSimple, FFmodelData, train_FF_model
from utils_FL import eval_FF_model, ServerNode
from utils_control import simulate_closed_loop_traj_follow

from paths import create_spline_paths
from plotting import plot_tracking_error_and_path

with open("config.json") as f:
    config = json.load(f)[0]

RNG_SEED    = config['simulation']['random_seed']

ff_input_train = config["NN_model"]["input_type_training"] # 'desired' or 'actual' for FF model input
ff_input_eval  = config["NN_model"]["input_type_eval"] # 'desired' or 'actual' for FF model input

pp = pprint.PrettyPrinter(indent=4, compact=True, width=40)
print(f"Configuration: ")
pp.pprint(config)

def sim_all_paths(paths, client=None, FF_type=None, FF_model=None, FF_input=None, cfg=None):
    """ Simulate paths with PI controller and the given FF_model. 
    
    Args:
    paths: list of paths
    FF_type: str, type of FF model {'model', 'analytic', None}
    FF_model: torch model, FF model - only necessary if FF_type='model'

    Returns:
    logs_FF_data: list of dataframes, data from the FF model
    logs_traj_data: list of dataframes, data from the trajectory

    """
    logs_traj_data = []
    for path in paths:
        init_state = {'x': path.x_fine[0], 'y': path.y_fine[0], 'psi': path.psi_fine[0], 'vl': path.v_long_profile[0]}
        car = BicycleModel(initial_state=init_state, cfg=cfg)
        traj = path.get_trajectory(cfg['simulation']['end_time'], dt=cfg["simulation"]["timestep"])
        log_traj_NN = simulate_closed_loop_traj_follow(traj, car, FF_type=FF_type, FF_model=FF_model, 
                                                       FF_input=FF_input, dt=cfg["simulation"]["timestep"])   
        logs_traj_data.append(log_traj_NN)
    return logs_traj_data

def sim_and_learn_centralized_FF(paths, FF_model=None, FF_type=None, FF_input=None, cfg=None):
    """ Simulate all paths given with the FF model and learn the FF model. """
    log_traj = sim_all_paths(paths, FF_type=FF_type, FF_model=FF_model, FF_input=FF_input, cfg=cfg)
    logs = [pd.DataFrame(log) for log in log_traj]
    df = pd.concat(logs)
    dataset = FFmodelData(df, FF_input=FF_input)
    dataloader = DataLoader(dataset, batch_size=cfg["NN_model"]["batch_size"], shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(FF_model.parameters(), lr=cfg["NN_model"]["learning_rate"])
    train_FF_model(FF_model, optimizer, dataloader, criterion, epochs=cfg["learning"]["local_epochs"])
    return

def main():
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    if os.getcwd().split('\\')[-1] == 'exp_models':
        path_file = 'paths.json'
    elif os.getcwd().split('\\')[-1] == 'experiments':
        path_file = 'exp_models\\paths.json'  
    PATHS = create_spline_paths(file=path_file)

    # create FF model
    FF_model = FFmodelSimple

    all_paths = list(range(len(PATHS)))
    test_path_idx = [2, 5, 8, 10]
    train_path_idx = [i for i in range(len(all_paths)) if i not in test_path_idx]
    train_paths = [PATHS[i] for i in train_path_idx]
    test_paths = [PATHS[i] for i in test_path_idx]
    # train_paths = train_paths[:2]

    results = []
    RMSEs = {}
    print(f"###"*20)

    global_client = ServerNode(id=0, 
                               model=FF_model(), 
                               dynamic_model='bicycle',
                               cfg=config)
    

    # initial simulation with PI controller
    print(f"Evaluate PI controller on test paths")
    res = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, 
                         path=path, cfg=config) for path in test_paths]
    RMSEs['PI only'] = [r[0] for r in res]

    print(f"###"*20)
    print(f"<<< Centralized Close-loop FF learning for {config['learning']['global_rounds']} epochs >>>")
    # simulate with PI + FF model
    for round in range(config["learning"]["global_rounds"]):
        print(f"Simulation with PI + FF model: Round {round+1}")
        sim_and_learn_centralized_FF(train_paths, FF_model=global_client.model, FF_type='model', 
                                     FF_input=ff_input_train, cfg=config)
        res = [eval_FF_model(FF_model=global_client.model, FF_type='model', 
                             FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]
        results.append(res)
        RMSEs['i'+str(round+1)] = [r[0] for r in res]
        print(f"###"*20)

    # add analytic FF results
    res = [eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, 
                         path=path, cfg=config) for path in test_paths]
    RMSEs['analytic FF'] = [r[0] for r in res]
    results.append(res)

    plot_tracking_error_and_path(RMSEs, test_paths, test_path_idx,
                                 title='Mean Tracking error by Path and Epoch: Centralized Learning')

    # evaluate the FF model on a new path
    path_idx = 3
    _, log_traj_NN = eval_FF_model(FF_model=global_client.model, FF_type='model', FF_input=ff_input_eval, path=test_paths[path_idx], cfg=config)
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=test_paths[path_idx],cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=test_paths[path_idx], cfg=config)

    plt.figure()
    plt.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='PI')
    plt.plot(log_traj_NN['x_a'], log_traj_NN['y_a'], 'b', label='PI + FF')
    plt.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'm', label='PI + FF (analytic)')
    plt.plot(test_paths[path_idx].x_fine, test_paths[path_idx].y_fine, 'k--', label='Desired')
    plt.legend()
    plt.grid()
    plt.title(test_paths[path_idx].name)
    plt.show()

    print(f"Plot FF control signal for Analytic and NN model for {test_paths[path_idx].name} paths!")
    plt.figure()
    plt.plot(log_traj_NN['time'], log_traj_NN['u_steer_FF'], 'b', label='FF')
    plt.plot(log_traj_analytic['time'], log_traj_analytic['u_steer_FF'], 'm', label='FF (analytic)')
    plt.xlabel('Time [s]')
    plt.ylabel('FF control signal')
    plt.legend()
    plt.grid()
    plt.title(test_paths[path_idx].name)

    print(f"Plot dpsi for analytic, PI, and PI+FF models for {test_paths[path_idx].name} paths!")
    plt.figure()
    plt.plot(log_traj_0['time'], log_traj_0['dpsi_desired'], 'r', label='PI - desired')
    plt.plot(log_traj_0['time'], log_traj_0['dpsi_actual'], 'r.', label='PI - actual')
    plt.plot(log_traj_NN['time'], log_traj_NN['dpsi_desired'], 'b', label='NN - desired')
    plt.plot(log_traj_NN['time'], log_traj_NN['dpsi_actual'], 'b.', label='NN - actual')
    plt.plot(log_traj_analytic['time'], log_traj_analytic['dpsi_desired'], 'm', label='analytic - desired')
    plt.plot(log_traj_analytic['time'], log_traj_analytic['dpsi_actual'], 'm.', label='analytic - actual')
    plt.xlabel('Time [s]')
    plt.ylabel('dpsi [rad/s]')
    plt.legend()
    plt.grid()
    plt.title(test_paths[path_idx].name)
    plt.show()

    print(f"Plot dpsi error for analytic, PI, and PI+FF models for all test paths!")
    fig = plt.figure(figsize=(10,10))
    for path_idx in range(0, len(test_paths)):
        _, log_traj_NN = eval_FF_model(FF_model=global_client.model, FF_type='model', FF_input=ff_input_eval, path=test_paths[path_idx], cfg=config)
        _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=test_paths[path_idx], cfg=config)
        _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=test_paths[path_idx], cfg=config)
        plt.subplot(len(test_paths),1,path_idx+1)
        plt.plot(log_traj_0['time'], log_traj_0['dpsi_desired'] - log_traj_0['dpsi_actual'], '.r', label='PI - error')
        plt.plot(log_traj_NN['time'], log_traj_NN['dpsi_desired'] - log_traj_NN['dpsi_actual'], 'db', label='Centralized Model - error')
        plt.plot(log_traj_analytic['time'], log_traj_analytic['dpsi_desired'] - log_traj_analytic['dpsi_actual'], 'xg', label='analytic - error')
        plt.xlabel('Time [s]')
        plt.ylabel('dpsi error [rad/s]')
        plt.legend()
        plt.ylim([-0.7, 0.5])
        plt.grid()
        plt.title(test_paths[path_idx].name)
    fig.suptitle("centralized model after 10 epochs")
    plt.show()

    pass


if __name__ == "__main__": 
    main() 