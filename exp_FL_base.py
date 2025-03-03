"""

This script performs the federated learning experiments. 

"""



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
from controller_models import FFmodelSimple

from paths import create_spline_paths

with open("config.json") as f:
    config = json.load(f)[0]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
client_idx = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

SAVE = True

RNG_SEED = config['simulation']['random_seed']

ff_input_train = config["NN_model"]["input_type_training"] # 'desired' or 'actual' for FF model input
ff_input_eval  = config["NN_model"]["input_type_eval"] # 'desired' or 'actual' for FF model input

global_rounds = config["learning"]["global_rounds"]


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


FFmodel = FFmodelSimple(n_neurons=config['NN_model']['hidden_layer_size'])

FFcentralized = FFmodel._create_copy()
# load state_dict form models/centralized_FFmodel.pth
FFcentralized.load_state_dict(torch.load('models/centralized_FFmodel.pth', weights_only=True), strict=True)
FFcentralized.eval()

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
# initial simulation with FB controller
print(f"###"*20)
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


if SAVE:
    torch.save(Server.model.state_dict(), 'models/federated_FFmodel.pth')


print(f"Learning done. ")
# add analytic FF results
res_FF = [eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=path, cfg=config) for path in test_paths]
res_FB = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=path, cfg=config) for path in test_paths]
res_NN_fed = [Server.eval_global_model(path=path, FF_type='model', FF_input=ff_input_eval) for path in test_paths]
res_NN_cent = [eval_FF_model(FF_model=FFcentralized, FF_type='model', FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]


# bar chart of the MTEs
MTEs_FF      = {client_idx[idx]: res_FF[i][0]      for i, idx in enumerate(test_path_idx)}
MTEs_FB      = {client_idx[idx]: res_FB[i][0]      for i, idx in enumerate(test_path_idx)}
MTEs_NN_fed  = {client_idx[idx]: res_NN_fed[i][0]  for i, idx in enumerate(test_path_idx)}
MTEs_NN_cent = {client_idx[idx]: res_NN_cent[i][0] for i, idx in enumerate(test_path_idx)}

print(f"Plot MTE for NN+FB, FF+FB, and FB only.")
plt.rcParams['font.size'] = 22
fig1, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MTEs_FF))  # the label locations
width = 0.45  # the width of the bars
bars_FB      = ax.bar(x - 1.5*width/4, MTEs_FB.values(), 0.1, label='FB', color='red')
bars_FF      = ax.bar(x - 0.5*width/4, MTEs_FF.values(), 0.1, label='FB + FF', color='green')
bars_NN_cent = ax.bar(x + 0.5*width/4, MTEs_NN_cent.values(), 0.1, label='FB + neural FF', color='purple')
bars_NN_fed  = ax.bar(x + 1.5*width/4, MTEs_NN_fed.values(), 0.1, label='FB + FL-based FF', color='magenta')

ax.set_ylabel('Mean Tracking Error')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(MTEs_FF.keys(), rotation=0)
ax.grid(axis='y')
ax.set_ylim(0, 0.5)
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')
# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncols=2)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
fig1.tight_layout()

if SAVE:
    fig1.savefig(f"img/plots_federated/testpaths_MTE.pdf")

print(f"Evaluate the FF model on all test paths!")
# 2x2 plot for each tets path
plt.rcParams['font.size'] = 14
fig2a, axs = plt.subplots(2, 1, figsize=(5, 10))
# loop over ax and plot a sinues in each plot
for (path_idx, pp), ax in zip(enumerate(test_paths[:2]), axs.flat):
    _, log_traj_NN_fed = eval_FF_model(FF_model=Server.model, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_NN_cent = eval_FF_model(FF_model=FFcentralized, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

    ax.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'green', label='FB + FF')
    ax.plot(log_traj_NN_cent['x_a'], log_traj_NN_cent['y_a'], 'purple', label='FB + neural FF')
    ax.plot(log_traj_NN_fed['x_a'], log_traj_NN_fed['y_a'], 'magenta', label='FB + FL-based FF')
    ax.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'green', linestyle='--', label='')

    ax.set_title(client_idx[test_path_idx[path_idx]] + ': ' + pp.name, color='black')
    ax.grid()
    ax.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
plt.tight_layout()

fig2b, axs = plt.subplots(2, 1, figsize=(5, 10))
plt.rcParams['font.size'] = 14
# loop over ax and plot a sinues in each plot
for (path_idx, pp), ax in zip(enumerate(test_paths[2:]), axs.flat):
    _, log_traj_NN_fed = eval_FF_model(FF_model=Server.model, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_NN_cent = eval_FF_model(FF_model=FFcentralized, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

    ax.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'green', label='FB + FF')
    ax.plot(log_traj_NN_cent['x_a'], log_traj_NN_cent['y_a'], 'purple', label='FB + neural FF')
    ax.plot(log_traj_NN_fed['x_a'], log_traj_NN_fed['y_a'], 'magenta', label='FB + FL-based FF')
    ax.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'green', linestyle='--', label='')

    ax.set_title(client_idx[test_path_idx[path_idx+2]] + ': ' + pp.name, color='black')
    ax.grid()
    ax.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
plt.tight_layout()

if SAVE:
    fig2a.savefig(f"img/plots_federated/testpath_solution_a.pdf")
    fig2b.savefig(f"img/plots_federated/testpath_solution_b.pdf")

print(f"Done.")


