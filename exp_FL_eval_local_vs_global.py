"""

In this experiment, we evaluate the number of local epochs vs. global rounds for the FL task at hand.

"""

# imports
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import torch

import json

from utils_FL import ClientNode, ServerNode
from controller_models import FFmodelSimple

from paths import create_spline_paths

with open("config.json") as f:
    config = json.load(f)[0]

SAVE = True
RERUN = False # experiment takes some time to run if RERUN = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
client_idx = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

RNG_SEED = config['simulation']['random_seed']

ff_input_train = config["NN_model"]["input_type_training"] # 'desired' or 'actual' for FF model input
ff_input_eval  = config["NN_model"]["input_type_eval"] # 'desired' or 'actual' for FF model input

torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# load paths
path_file = 'paths.json'
PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])

test_path_idx = config['learning']['test_path_idx']
if len(test_path_idx) == 0:
    test_path_idx = [0, 5, 7, 10]
train_path_idx = [i for i in range(len(PATHS)) if i not in test_path_idx]

train_paths = [PATHS[i] for i in train_path_idx]
test_paths = [PATHS[i] for i in test_path_idx]

# 

def run_FL(global_rounds=1, local_epochs=1, config=config, 
           paths=PATHS, nr_test_paths=4, repeat=1, seed=None):

    print(f"*****"*20)
    print(f"FL simulation with {global_rounds} global rounds and {local_epochs} local epochs.")
    print(f"*****"*20)
    
    config['learning']['local_epochs'] = local_epochs
    config['learning']['global_rounds'] = global_rounds

    results = {}

    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    for r in range(repeat):
        print(f"Repeat {r+1} ...")
        nr_paths = len(paths)
        test_path_idx = np.random.choice(nr_paths, nr_test_paths, replace=False)
        train_path_idx = [i for i in range(nr_paths) if i not in test_path_idx]

        train_paths = [paths[i] for i in train_path_idx]
        test_paths = [paths[i] for i in test_path_idx]

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

        print(f"<<< Federated Close-loop FF learning for {global_rounds} epochs >>>")
        average_MTE = {}
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
            average_MTE['Round_' + str(round+1)] = np.mean([r[0] for r in res])
        average_MTE['Test path idxs'] = test_path_idx
        
        results[f"Run {r+1}"] = average_MTE

    df = pd.DataFrame(results.values())

    print(f"Learning done. ")
    return df

if RERUN:
    print(f"Running the FL simulation ...")
    G = 30
    repeat = 10
    df_R10_E01 = run_FL(global_rounds=G, local_epochs=1,  repeat=repeat, seed=RNG_SEED)
    df_R10_E02 = run_FL(global_rounds=G, local_epochs=2,  repeat=repeat, seed=RNG_SEED)
    df_R10_E05 = run_FL(global_rounds=G, local_epochs=5,  repeat=repeat, seed=RNG_SEED)
    df_R10_E10 = run_FL(global_rounds=G, local_epochs=10, repeat=repeat, seed=RNG_SEED)
else:
    df_R10_E01 = pd.read_csv("results/FL_local_vs_global/R10_E01.csv", index_col=0)
    df_R10_E02 = pd.read_csv("results/FL_local_vs_global/R10_E02.csv", index_col=0)
    df_R10_E05 = pd.read_csv("results/FL_local_vs_global/R10_E05.csv", index_col=0)
    df_R10_E10 = pd.read_csv("results/FL_local_vs_global/R10_E10.csv", index_col=0)
if SAVE:
    df_R10_E01.to_csv("results/FL_local_vs_global/R10_E01.csv")
    df_R10_E02.to_csv("results/FL_local_vs_global/R10_E02.csv")
    df_R10_E05.to_csv("results/FL_local_vs_global/R10_E05.csv")
    df_R10_E10.to_csv("results/FL_local_vs_global/R10_E10.csv")


print(f"Simulations done --> plot results... ")

# save the results
cols = df_R10_E01.columns[:-1]

mte_R10_E01_mean = df_R10_E01[cols].mean()
mte_R10_E01_std  = df_R10_E01[cols].std()
mte_R10_E02_mean = df_R10_E02[cols].mean()
mte_R10_E02_std  = df_R10_E02[cols].std()
mte_R10_E05_mean = df_R10_E05[cols].mean()
mte_R10_E05_std  = df_R10_E05[cols].std()
mte_R10_E10_mean = df_R10_E10[cols].mean()
mte_R10_E10_std  = df_R10_E10[cols].std()


# Prepare data for plotting
E1_col = 'blue'
E2_col = 'green'
E5_col = 'red'
E10_col = 'magenta'
width = 0.45
e1, e2, e5, e10, e15, e20, e25, e30 = 'E1', 'E2', 'E5', 'E10', 'E15', 'E20', 'E25', 'E30'

# bar chart of the MTEs
fig, ax = plt.subplots(figsize=(10, 5))

for xpos, GR in enumerate([1, 2, 5, 10, 15, 20, 25, 30]):

    ax.bar(     xpos - 1.5*width/4, mte_R10_E01_mean[f"Round_{GR}"], 0.1, label=e1, color=E1_col)
    ax.errorbar(xpos - 1.5*width/4, mte_R10_E01_mean[f"Round_{GR}"], yerr=mte_R10_E01_std[f"Round_{GR}"], fmt='none', ecolor='black', capsize=5)

    ax.bar(     xpos - 0.5*width/4, mte_R10_E02_mean[f"Round_{GR}"], 0.1, label=e2, color=E2_col)
    ax.errorbar(xpos - 0.5*width/4, mte_R10_E02_mean[f"Round_{GR}"], yerr=mte_R10_E02_std[f"Round_{GR}"], fmt='none', ecolor='black', capsize=5)

    ax.bar(     xpos + 0.5*width/4, mte_R10_E05_mean[f"Round_{GR}"], 0.1, label=e5, color=E5_col)
    ax.errorbar(xpos + 0.5*width/4, mte_R10_E05_mean[f"Round_{GR}"], yerr=mte_R10_E05_std[f"Round_{GR}"], fmt='none', ecolor='black', capsize=5)

    # ax.bar(     xpos + 1.5*width/4, mte_R10_E10_mean[f"Round_{GR}"], 0.1, label=e10, color=E10_col)
    # ax.errorbar(xpos + 1.5*width/4, mte_R10_E10_mean[f"Round_{GR}"], yerr=mte_R10_E10_std[f"Round_{GR}"], fmt='none', ecolor='black', capsize=5)

    e1, e2, e5, e10, e15, e20, e25, e30 = '','','','','','','','' # only label the first time

ax.set_xlabel('Global Rounds')
ax.set_ylabel('Average Mean Tracking Error')
# ax.set_title('Federated Learning Performance')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xticklabels(['1', '2', '5', '10', '15', '20', '25', '30'])
plt.legend()
plt.ylim(0, 0.5)    
plt.grid()
plt.tight_layout()

if SAVE:
    fig.savefig(f"img/plots_federated/fl_local_vs_global_bar.pdf")

# line chart of the MTEs
plt.rcParams['font.size'] = 22

fig2, ax = plt.subplots(figsize=(10, 5))
alpha = 0.1
x_vals = np.arange(1, mte_R10_E01_std.shape[0]+1)
ax.plot(x_vals, mte_R10_E01_mean, label='E=1', color=E1_col)
ax.fill_between(x_vals, mte_R10_E01_mean - mte_R10_E01_std, mte_R10_E01_mean + mte_R10_E01_std, color=E1_col, alpha=alpha)

ax.plot(x_vals, mte_R10_E02_mean, label='E=2', color=E2_col)
ax.fill_between(x_vals, mte_R10_E02_mean - mte_R10_E02_std, mte_R10_E02_mean + mte_R10_E02_std, color=E2_col, alpha=alpha)

ax.plot(x_vals, mte_R10_E05_mean, label='E=5', color=E5_col)
ax.fill_between(x_vals, mte_R10_E05_mean - mte_R10_E05_std, mte_R10_E05_mean + mte_R10_E05_std, color=E5_col, alpha=alpha)

# ax.plot(x_vals, mte_R10_E10_mean, label='E10', color=E10_col)
# ax.fill_between(x_vals, mte_R10_E10_mean - mte_R10_E10_std, mte_R10_E10_mean + mte_R10_E10_std, color=E10_col, alpha=alpha)

ax.set_xlabel('Global Communication Rounds')
ax.set_ylabel('Average Mean Tracking Error')


# ax.set_xticklabels(['1', '2', '5', '10', '15', '20', '25', '30'])
# ax.set_yscale('log')
ax.grid()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.legend()
plt.tight_layout()

if SAVE:
    fig2.savefig(f"img/plots_federated/fl_local_vs_global_line.pdf")

print(f"Done.")