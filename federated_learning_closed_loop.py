"""

Federated learning with an application of the learned FF model. 


"""

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import colormaps as cmaps # type: ignore

import os
import json
import pprint 

from utils_FL import ClientNode, ServerNode, eval_FF_model
from controller_models import FFmodelSimple, FFmodelResNet

from plotting import plot_tracking_error_and_path
from paths import create_spline_paths

with open("config.json") as f:
    config = json.load(f)[0]

RNG_SEED = config['simulation']['random_seed']

ff_input_train = config["NN_model"]["input_type_training"] # 'desired' or 'actual' for FF model input
ff_input_eval  = config["NN_model"]["input_type_eval"] # 'desired' or 'actual' for FF model input

global_rounds = config["learning"]["global_rounds"]

pp = pprint.PrettyPrinter(indent=4, compact=True, width=40)
print(f"Configuration: ")
pp.pprint(config)

def main():
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    if os.getcwd().split('\\')[-1] == 'paper_FedFF':
        path_file = 'paths.json'

    PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])

    test_path_idx = config['learning']['test_path_idx']
    if len(test_path_idx) == 0:
        test_path_idx = [0, 5, 7, 10]
    train_path_idx = [i for i in range(len(PATHS)) if i not in test_path_idx]
    train_paths = [PATHS[i] for i in train_path_idx]
    test_paths = [PATHS[i] for i in test_path_idx]
    # train_paths = train_paths[:2]

    # plot_paths(train_paths, title='Train paths')
    # plot_paths(test_paths, title='Test paths')

    # FFmodel = FFmodelResNet
    FFmodel = FFmodelSimple(n_neurons=config['NN_model']['hidden_layer_size'])

    Server = ServerNode(id=-1, 
                        model=FFmodel._create_copy(),
                        cfg=config, 
                        dynamic_model='bicycle')
    ClientNodes = [ClientNode(id=i+1, 
                              model=FFmodel._create_copy(), 
                              dynamic_model='bicycle',
                              cfg=config, 
                              path=train_paths[i]) for i in range(len(train_paths))]

    results = []
    RMSEs = {}
    # initial simulation with FB controller
    res = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=path, cfg=config) for path in test_paths]
    RMSEs['FB only'] = [r[0] for r in res]
    print(f"###"*20)
    LGCs = []
    print(f"<<< Federated Close-loop FF learning for {global_rounds} epochs >>>")
    for round in range(global_rounds):
        print(f"Round {round+1}: local client simulation and learning ...")
    
        # local client simulation and learning
        for i, (client, path) in enumerate(zip(ClientNodes, train_paths)):
            print(f"\tLocal {client} working ...")
            client.receive_global_model(Server.model)
            client.simulate_and_learn_FF_local(FF_input=ff_input_train, FF_type='model')

        # gather client messages
        client_msgs = [client.send_model_to_server() for client in ClientNodes]
        Server.fedavg(client_msgs)

        print(f"Round {round+1}: Evaluate FB controller + global FF on test paths")
        res = [Server.eval_global_model(path=path, FF_type='model', FF_input=ff_input_eval) for path in test_paths]
        results.append(res)
        RMSEs['i'+str(round+1)] = [r[0] for r in res]

        # calculate LGC
        lcg, lcg_matrix = Server.calc_LCG(client_messages=client_msgs)
        LGCs.append(lcg)
        print(f"###"*20)
    
    # plt.figure()
    # plt.plot(range(global_rounds), LGCs, 'o-')
    # plt.xlabel('Epochs')
    # plt.ylabel('LGC')
    # Server.plot_LGC_matrix(round_number=1)
    # Server.plot_LGC_matrix(round_number=global_rounds)

    # add analytic FF results
    res = [eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=path, cfg=config) for path in test_paths]
    RMSEs['analytic'] = [r[0] for r in res]
    results.append(res)
    
    plot_tracking_error_and_path(RMSEs, test_paths, test_path_idx, title='Mean Tracking error by Path and Epoch: Federated Learning')
    
    # evaluate the FF model on a new path
    path_idx = 0
    pp = test_paths[path_idx]
    # pp = train_paths[path_idx]
    _, log_traj_NN = eval_FF_model(FF_model=Server.model, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

    print(f"Plot the trajectory for Analytic and NN model for {pp.name} paths!")
    plt.figure()
    plt.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
    plt.plot(log_traj_NN['x_a'], log_traj_NN['y_a'], 'b', label='FB + FF (NN)')
    plt.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'm', label='FB + FF (analytic)')
    plt.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')
    plt.legend()
    plt.grid()
    plt.title(pp.name)
    # Adjust layout to prevent legend from being cut off
    plt.show()


    print(f"Plot steering control signal for Analytic and NN model for {test_paths[path_idx].name} paths!")
    plt.figure()
    plt.plot(log_traj_0['time'], log_traj_0['u_steer'], 'r', label='FB only')
    plt.plot(log_traj_NN['time'], log_traj_NN['u_steer'], 'g', label='FB + FF (NN)')
    plt.plot(log_traj_analytic['time'], log_traj_analytic['u_steer'], '--m', label='FB + FF (analytic)')
    plt.xlabel('Time [s]')
    plt.ylabel('Steering control signal')
    plt.legend()
    plt.grid()
    plt.title(pp.name)

    print(f"Evaluate the FF model on all test paths!")
    # 2x2 plot for each tets path
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # loop over ax and plot a sinues in each plot
    for (path_idx, pp), ax in zip(enumerate(test_paths), axs.flat):
        _, log_traj_NN = eval_FF_model(FF_model=Server.model, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
        _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
        _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

        ax.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
        ax.plot(log_traj_NN['x_a'], log_traj_NN['y_a'], 'b', label='FB + neural FF')
        ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'm', label='FB + FF (analytic)')
        ax.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')

        ax.set_title(pp.name)
        ax.grid()
        ax.legend()
        plt.pause(0.25)

    plt.tight_layout()

    print(f"Plot x-y position for analytic, FB, and FB+FF models for {pp.name} paths!")
    tt = pp.get_trajectory(T_end=config['simulation']['end_time'], dt=config['simulation']['timestep'])
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(log_traj_0['time'], log_traj_0['x_a'], 'r', label='FB - x_a', linewidth=2)
    plt.plot(log_traj_analytic['time'], log_traj_analytic['x_a'], 'm', label='analytic - x_a', linewidth=2)
    plt.plot(log_traj_NN['time'], log_traj_NN['x_a'], 'b', label='FL NN - xa')
    plt.plot(tt['time'], tt['x'], 'k--', label='x_d')
    plt.xlabel('Time [s]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid()
    plt.title(pp.name)
    plt.subplot(2,1,2)
    plt.plot(log_traj_0['time'], log_traj_0['y_a'], 'r', label='FB - y_a', linewidth=2)
    plt.plot(log_traj_analytic['time'], log_traj_analytic['y_a'], 'm', label='analytic - y_a', linewidth=2)
    plt.plot(log_traj_NN['time'], log_traj_NN['y_a'], 'b', label='FL NN - y_a')
    plt.plot(tt['time'], tt['y'], 'k--', label='y_d')
    plt.xlabel('Time [s]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Plot trajectory error error for analytic, FB, and FB+FF models for all test paths!")
    fig = plt.figure(figsize=(10,10))
    plt.rcParams['font.size'] = 12
    for path_idx in range(0, len(test_paths)):
        _, log_traj_FL = eval_FF_model(FF_model=Server.model, FF_type='model', 
                                       FF_input=ff_input_eval, path=test_paths[path_idx], cfg=config)
        _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=test_paths[path_idx], cfg=config)
        _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=test_paths[path_idx], cfg=config)
        plt.subplot(len(test_paths),1,path_idx+1)
        plt.plot(log_traj_0['time'], calc_trajectory_error(log_traj_0), '.r', label='FB - error')
        plt.plot(log_traj_FL['time'], calc_trajectory_error(log_traj_FL), 'm', label='FL Model - error')
        plt.plot(log_traj_analytic['time'], calc_trajectory_error(log_traj_analytic), 'xg', label='analytic - error')
        # plt.plot(results[-2][path_idx][1]['time'], results[-2][path_idx][1]['dpsi_desired'] - results[-1][path_idx][1]['dpsi_actual'], 'og', label='epoch 9 - error')
        plt.xlabel('Time [s]')
        plt.ylabel('Trajectory error [m]')
        plt.legend()
        plt.ylim([-0.5, 0.5]) 
        plt.grid()
        plt.title(test_paths[path_idx].name)
    fig.suptitle(f"Federated model after {global_rounds} epochs")
    plt.tight_layout()
    plt.show()

    return

def calc_trajectory_error(log_traj):
    e_x = (log_traj['x_d'] - log_traj['x_a']).values
    e_y = (log_traj['y_d'] - log_traj['y_a']).values
    trajectory_error = (e_x**2 + e_y**2)**0.5
    return trajectory_error

def plot_FF_model(model, ax=None):

    x1 = np.linspace(-1.5, 1.5, 20) # kappa
    x2 = np.linspace(0, 2, 20) # v
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = model(torch.tensor(X, dtype=torch.float32)).detach().numpy().reshape(X1.shape)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Curvature')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('u_steer')
    ax.plot_surface(X1, X2, y, cmap='viridis', label='NN')
    plt.show()


def plot_paths(PATHS, title=''):

    n_paths = len(PATHS)

    n_cols = 3
    n_rows = np.ceil(n_paths / n_cols).astype(int)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.ravel()
    for i, path in enumerate(PATHS):
        ax = axs[i]
        # ax.plot(path.x_fine, path.y_fine, 'k--', label="Path")
        s = ax.scatter(path.x_fine, path.y_fine, s=1, 
                       c=path.v_long_profile, cmap=cmaps.cmp_b2r, vmin=0, vmax=2)
        ax.scatter(path.waypoints[:, 0], path.waypoints[:, 1], c="r", label="Waypoints")
        ax.set_title(path.name)
        ax.set_xlim([-12,12])
        ax.set_ylim([-12, 12])
        # ax.axis("equal")
        ax.grid()

    fig.colorbar(s, ax=ax, label='Desired Velocity [m/s]')
    fig.suptitle(title)
    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=False)

if __name__ == "__main__": 
    main() 