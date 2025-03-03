"""

In this experiment, we train the local neural FF controller, i.e., for each client
we learn a separate NN model. 

We evaluate the performance on the test paths.

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.utils.data import DataLoader

import os
import json
import pickle

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

if os.getcwd().split('\\')[-1] == 'paper_FedFF':
    path_file = 'paths.json'

PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])

test_path_idx = config['learning']['test_path_idx']
if len(test_path_idx) == 0:
    test_path_idx = [0, 5, 7, 10]
train_path_idx = [i for i in range(len(PATHS)) if i not in test_path_idx]

train_paths = [PATHS[i] for i in train_path_idx]
test_paths = [PATHS[i] for i in test_path_idx]

# FFmodel 
FFmodel = FFmodelSimple(n_neurons=config['NN_model']['hidden_layer_size'])
local_models =  {idx: FFmodel._create_copy() for idx in train_path_idx}

for path, idx in zip(train_paths, train_path_idx):
    print(f"Training local neural FF for client {client_idx[idx]}")
    # simulate all training paths and store the data
    res = [eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=path, cfg=config)]

    df = pd.concat([r[1] for r in res], ignore_index=True)
    optimizer = torch.optim.Adam(local_models[idx].parameters(), lr=config['NN_model']['learning_rate'])
    loss_fn = nn.MSELoss()
    dataset = FFmodelData(df, FF_input=ff_input_train)
    trainloader = DataLoader(dataset, 
                            batch_size=config['NN_model']['batch_size'], 
                            shuffle=True)
    nr_samples = len(trainloader.dataset)

    print(f"\tTraining the FF model.")
    train_losses = []
    local_models[idx].train()
    for epoch in range(config['learning']['global_rounds']):
        train_loss = []
        for i, (X, y) in enumerate(trainloader):
            optimizer.zero_grad()
            y_pred = local_models[idx](X).flatten()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_losses.append(np.mean(train_loss))
        print(f"\t\tEpoch: {epoch} | Loss: {train_losses[epoch]**0.5:.6f} |") # loss is MSE, print RMSE
    print(f"Training done for client {client_idx[idx]}")

if SAVE:
    # save list of local models to pickle
    with open('models/local_models.pkl', 'wb') as f:
        pickle.dump(local_models, f)

# load centralized and FL model
FFcentralized = FFmodel._create_copy()
# load state_dict form models/centralized_FFmodel.pth
FFcentralized.load_state_dict(torch.load('models/centralized_FFmodel.pth', weights_only=True), strict=True)
FFcentralized.eval()

FFfederated = FFmodel._create_copy()
# load state_dict form models/federated_FFmodel.pth
FFfederated.load_state_dict(torch.load('models/federated_FFmodel.pth', weights_only=True), strict=True)
FFfederated.eval()

print(f"Evaluate the centralized FF model on all test paths!")
res_centralized = [eval_FF_model(FF_model=FFcentralized, FF_type='model', FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]
print(f"Evaluate the federated FF model on all test paths!")
res_federated = [eval_FF_model(FF_model=FFfederated, FF_type='model', FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]

print(f"Evaluate the FF model on all test paths!")
res_local = {}
for idx in train_path_idx:
    res_local[str(idx)] = [eval_FF_model(FF_model=local_models[idx], FF_type='model', FF_input=ff_input_eval, path=path, cfg=config) for path in test_paths]

df_res = pd.DataFrame({client_idx[idx]: res_centralized[i][0] for i, idx in enumerate(test_path_idx)}, index=['Centralized'])
df_res.loc['Federated'] = {client_idx[idx]: res_federated[i][0] for i, idx in enumerate(test_path_idx)}

for k, v in res_local.items():
    df_res.loc[f'Local {client_idx[int(k)]}'] = {client_idx[idx]: res_local[k][i][0] for i, idx in enumerate(test_path_idx)}


# Plot the results in a bar chart
print(f"Plot MTEs for local models!")
fig1, ax = plt.subplots(figsize=(12, 6))
df_res.T.plot(kind='bar', ax=ax)
ax.set_ylabel('Mean Tracking Error')
ax.set_xlabel('Test Clients')
ax.legend(title='Model Type', ncol=2)
ax.grid(axis='y')
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')
ax.set_xticklabels(df_res.columns, rotation=0)

ax.set_ylim(0, 0.5)
# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=2)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
fig1.tight_layout()

if SAVE:
    fig1.savefig(f"img/plots_local/model_comparison.pdf")


df_res_2 = df_res.copy()
df_res_2.drop(df_res_2.index[0], inplace=True)
df_res_2 -= df_res_2.iloc[0]

print(f"Plot MTEs - federated MTE!")
fig2, ax = plt.subplots(figsize=(12, 6))
df_res_2.iloc[1:].T.plot(kind='bar', ax=ax)
ax.set_ylabel(r'$\text{MTE}_{local} - \text{MTE}_{fed}$')
ax.set_xlabel('Test Clients')
ax.legend(title='Model Type', ncol=2)
ax.grid(axis='y')
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')
ax.set_xticklabels(df_res.columns, rotation=0)

ax.set_ylim(-0.075, 0.5)
# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=2)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
fig2.tight_layout()

if SAVE:
    fig2.savefig(f"img/plots_local/model_comparison_baselined.pdf")

df_res_3 = df_res.copy()
df_res_3.drop(df_res_3.index[0], inplace=True)
df_res_3 /= df_res_3.iloc[0]

print(f"Plot MTEs - federated MTE!")
plt.rcParams['font.size'] = 22
fig3, ax = plt.subplots(figsize=(10, 5))
df_res_3.iloc[1:].T.plot(kind='bar', ax=ax)
# ax.set_ylabel(r'$\frac{\text{MTE}_{local}}{\text{MTE}_{fed}}$')
# ax.set_ylabel(r'$\text{MTE}_{local} \hspace{0.25} / \hspace{0.25} \text{MTE}_{fed}$')
ax.set_ylabel('MTE-local / MTE-fed')

ax.set_xlabel('Test Clients')
ax.legend(title='Model Type', ncol=2)
ax.grid(axis='y')
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')
ax.set_xticklabels(df_res.columns, rotation=0)

# ax.set_yscale('log')
plt.axhline(y=1, color='r', linestyle='--')# ax.set_ylim(-0.075, 0.5)
# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=2)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
fig3.tight_layout()

if SAVE:
    fig3.savefig(f"img/plots_local/model_comparison_ratio.pdf")


print(f"Done.")