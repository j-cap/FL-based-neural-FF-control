"""

In this exeriment, we use the train-test split of the clients (given in config.json)
to learn a neural FF controller. We then simulate the KBM model with the learned
controller and compare the performance with the analytic FF+FB controller.


"""

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader

import os
import json

from utils_FL import eval_FF_model
from controller_models import FFmodelSimple, FFmodelData

from paths import create_spline_paths

with open("config.json") as f:
    config = json.load(f)[0]

SAVE = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
client_idx = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

client_names = ['Left Turn Dominant Egg', 'Left Turn Dominant Egg Slow', 'Left Turn Dominant Egg Fast', 
                'Left Turn Dominant Ditched Ellipsoid', 'Left Turn Dominant Circle', 
                'Right Balanced Figure 8', 'Let Turn Dominant Potato', 'Right Turn Dominant Potato',
                'Right Unbalanced Figure Eight', 'Right Turn Dominant Ditched Circe', 
                'Right Turn Dominant Circle', 'Right Turn Dominant Ditched Circle Large']


RNG_SEED = config['simulation']['random_seed']

ff_input_train = config["NN_model"]["input_type_training"] # 'desired' or 'actual' for FF model input
ff_input_eval  = config["NN_model"]["input_type_eval"] # 'desired' or 'actual' for FF model input

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# if os.getcwd().split('\\')[-1] == 'paper_FedFF':
path_file = 'paths.json'

PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])

test_path_idx = config['learning']['test_path_idx']
if len(test_path_idx) == 0:
    test_path_idx = [0, 5, 7, 10]
train_path_idx = [i for i in range(len(PATHS)) if i not in test_path_idx]

train_paths = [PATHS[i] for i in train_path_idx]
test_paths = [PATHS[i] for i in test_path_idx]

# FFmodel = FFmodelResNet
FFmodel = FFmodelSimple(n_neurons=config['NN_model']['hidden_layer_size'])

# simulate all training paths and store the data
print(f"Training the FF model.")
for epoch in range(config['learning']['global_rounds']):
    if epoch == 0:
        res_train_paths = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=path, cfg=config) for path in train_paths]
    else:
        res_train_paths = [eval_FF_model(FF_model=FFmodel, FF_type='model', FF_input=ff_input_train, path=path, cfg=config) for path in train_paths]

    df = pd.concat([r[1] for r in res_train_paths], ignore_index=True)

    optimizer = torch.optim.Adam(FFmodel.parameters(), lr=config['NN_model']['learning_rate'])
    loss_fn = nn.MSELoss()
    # print(f"\tTraining the FF model for client {id}")
    dataset = FFmodelData(df, FF_input=ff_input_train)
    trainloader = DataLoader(dataset, 
                            batch_size=config['NN_model']['batch_size'], 
                            shuffle=True)
    # nr_samples = len(trainloader.dataset)

    train_losses = []
    FFmodel.train()
    # for epoch in range(config['learning']['global_rounds']):
    train_loss = []
    for i, (X, y) in enumerate(trainloader):
        optimizer.zero_grad()
        y_pred = FFmodel(X).flatten()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_losses.append(np.mean(train_loss))
    print(f"\t\tEpoch: {epoch} | Loss: {train_losses[0]**0.5:.6f} |") # loss is MSE, print RMSE

if SAVE:
    torch.save(FFmodel.state_dict(), 'models/centralized_FFmodel.pth')

print(f"Evaluate the FF model on all test paths!")
res_NN_test_paths = [eval_FF_model(FF_model=FFmodel, FF_type='model', FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]
res_FF_test_paths = [eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=path, cfg=config) for path in test_paths]
res_FB_test_paths = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=path, cfg=config) for path in test_paths]

MTEs_NN = {client_idx[idx]: res_NN_test_paths[i][0] for i, idx in enumerate(test_path_idx)}
MTEs_FF = {client_idx[idx]: res_FF_test_paths[i][0] for i, idx in enumerate(test_path_idx)}
MTEs_FB = {client_idx[idx]: res_FB_test_paths[i][0] for i, idx in enumerate(test_path_idx)}

print(f"Plot MTE for NN+FB, FF+FB, and FB only.")
fig1, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MTEs_FF))  # the label locations
width = 0.35  # the width of the bars
bars_FB = ax.bar(x - width/3, MTEs_FB.values(), 0.1, label='FB', color='red')
bars_FF = ax.bar(x          , MTEs_FF.values(), 0.1, label='FB + FF', color='green')
bars_NN = ax.bar(x + width/3, MTEs_NN.values(), 0.1, label='FB + neural FF', color='purple')

ax.set_ylabel('Mean Tracking Error')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(MTEs_FF.keys(), rotation=0)
ax.legend()
ax.grid()
ax.set_ylim(0, 0.5)
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')
# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
fig1.tight_layout()
plt.show()

if SAVE:
    fig1.savefig(f"img/plots_centralized/testpaths_MTE.pdf")

print(f"Evaluate the FF model on all test paths!")
# 2x2 plot for each tets path
fig2, axs = plt.subplots(2, 2, figsize=(10, 10))
# loop over ax and plot a sinues in each plot
for (path_idx, pp), ax in zip(enumerate(test_paths), axs.flat):
    _, log_traj_NN = eval_FF_model(FF_model=FFmodel, FF_type='model', FF_input=ff_input_eval, path=pp, cfg=config)
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

    ax.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'green', label='FB + FF')
    ax.plot(log_traj_NN['x_a'], log_traj_NN['y_a'], 'purple', label='FB + neural FF')
    ax.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')

    ax.set_title(client_idx[test_path_idx[path_idx]] + ': ' + client_names[path_idx], color='red')
    ax.grid()
    ax.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    for spine in ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

plt.tight_layout()

if SAVE:
    fig2.savefig(f"img/plots_centralized/testpath_solution.pdf")

print(f"Done.")