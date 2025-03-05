"""

This script simulates the KBM model with a FF controller and a FB controller.

Produces the following plots:
- Bar chart: Mean tracking error for each path
- Line chart: t vs. x_d(t), x_a(t), with error as shaded region in a subplot
- Line chart: t vs. y_d(t), y_a(t), with error as shaded region in a subplot

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from paths import create_spline_paths
from utils_FL import eval_FF_model

SAVE = True   

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
client_idx = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

with open("config.json") as f:
    config = json.load(f)[0]

path_file = 'paths.json'

RNG_SEED = config['simulation']['random_seed']

PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])

test_path_idx = config['learning']['test_path_idx']
if len(test_path_idx) == 0:
    test_path_idx = [0, 5, 7, 10]
train_path_idx = [i for i in range(len(PATHS)) if i not in test_path_idx]
train_paths = [PATHS[i] for i in train_path_idx]
test_paths = [PATHS[i] for i in test_path_idx]

print(f"Simulate KBM model with FB controller.")
res_FB = [eval_FF_model(FF_model=None, 
                     FF_type=None, 
                     FF_input=None, path=path, cfg=config) for path in PATHS]
MTEs = {client_idx[i]: res_FB[i][0] for i, path in enumerate(PATHS)}

print(f"Simulate KBM model with FB + FF controller.")
res_FF = [eval_FF_model(FF_model=None, 
                        FF_type='analytic', 
                        FF_input=None, path=path, cfg=config) for path in PATHS]
MTEs_FF = {client_idx[i]: res_FF[i][0] for i, path in enumerate(PATHS)}


print(f"Plotting the mean tracking error for each path.")
# Define colors for bars
colors_FB = ['red' if path.name in [p.name for p in test_paths] else 'blue' for path in PATHS]
colors_FF = ['green' for _ in PATHS]

# bar chart of the MTEs with one bar per path
x = np.arange(len(MTEs))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5))
bars_FB = ax.bar(x - width/2, MTEs.values(), width, label='FB Test Clients', color=colors_FB)
bars_FF = ax.bar(x + width/2, MTEs_FF.values(), width, label='FB + FF', color=colors_FF)
ax.set_ylabel('Mean Tracking Error')
# ax.set_title('Mean Tracking Error by Path with FB + FF Controller')
ax.set_xticks(x)
ax.set_xticklabels(MTEs.keys(), rotation=0)
ax.legend()
ax.grid(axis='y')
ax.set_ylim(0, 0.5)
# Customize xtick marks
ax.tick_params(axis='x', which='both', length=5, color='lightgrey')

# Add another entry to the legend
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.bar([0], [0], color='blue', lw=8))
labels.append('FB Train Clients')
ax.legend(handles, labels)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

fig.tight_layout()
plt.show()

if SAVE:
    fig.savefig('img/plots_ctrl_performance/MTE_comparison.pdf')

# Plot the paths with the FF controller
print(f"Plotting the X paths with the FF controller.")
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1, 'wspace': 0.3175}, sharex='col')
for idx, path in enumerate(PATHS):
    x_error_FB = np.cos(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['x_d'] - res_FB[idx][1]['x_a']) + \
                    np.sin(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['y_d'] - res_FB[idx][1]['y_a'])
    x_error_FF = np.cos(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['x_d'] - res_FF[idx][1]['x_a']) + \
                    np.sin(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['y_d'] - res_FF[idx][1]['y_a'])
    y_error_FB = np.sin(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['x_d'] - res_FB[idx][1]['x_a']) - \
                    np.cos(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['y_d'] - res_FB[idx][1]['y_a'])
    y_error_FF = np.sin(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['x_d'] - res_FF[idx][1]['x_a']) - \
                    np.cos(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['y_d'] - res_FF[idx][1]['y_a'])
    
    if path.name == 'right_turn_dominant_circle':
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_d'], 'k--', label='Desired')
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_a'], 'b', label='FB')
        ax[0,0].plot(res_FF[idx][1]['time'], res_FF[idx][1]['x_a'], 'r', label='FB + FF')
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_d'], 'k--', label='')
        ax[0,0].set_ylabel('X [m]')
        ax[0,0].grid()
        ax[0,0].set_ylim(-2, 4)
        ax[0,0].set_yticks(np.arange(-2, 3.5, 1))
        ax[0,0].legend()

        ax[1,0].plot(res_FB[idx][1]['time'], x_error_FB, 'b', label='FB')
        ax[1,0].plot(res_FF[idx][1]['time'], x_error_FF, 'r', label='FB + FF')
        ax[1,0].grid()
        ax[1,0].set_xlabel('Time [s]')
        ax[1,0].set_ylabel(r'$\epsilon_x$ [m]')
        # ax[1,0].set_yticks(np.arange(-0.49, 0.5, 0.25))
        ax[1,0].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax[1,0].set_ylim(-0.55, 0.55)

        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_d'], 'k--', label='Desired')
        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_a'], 'b', label='FB')
        ax[0,1].plot(res_FF[idx][1]['time'], res_FF[idx][1]['y_a'], 'r', label='FB + FF')
        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_d'], 'k--', label='')

        ax[0,1].grid() 
        ax[0,1].set_ylabel('Y [m]')
        ax[0,1].set_ylim(-2, 4)
        ax[0,1].set_yticks(np.arange(-2, 3.5, 1))
        ax[0,1].legend()

        ax[1,1].plot(res_FB[idx][1]['time'], y_error_FB, 'b', label='FB')
        ax[1,1].plot(res_FF[idx][1]['time'], y_error_FF, 'r', label='FB + FF')
        ax[1,1].grid()
        ax[1,1].set_xlabel('Time [s]')
        ax[1,1].set_ylabel(r'$\epsilon_y$ [m]')
        ax[1,1].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax[1,1].set_ylim(-0.55, 0.55)

ax[0,0].label_outer()
for ax in ax.flat:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
if SAVE:
    fig.savefig('img/plots_ctrl_performance/tracking_error_clients_XI.pdf')

print(f"Plotting the X paths with the FF controller.")
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1, 'wspace': 0.3175}, sharex='col')
for idx, path in enumerate(PATHS):
    x_error_FB = np.cos(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['x_d'] - res_FB[idx][1]['x_a']) + \
                    np.sin(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['y_d'] - res_FB[idx][1]['y_a'])
    x_error_FF = np.cos(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['x_d'] - res_FF[idx][1]['x_a']) + \
                    np.sin(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['y_d'] - res_FF[idx][1]['y_a'])
    y_error_FB = np.sin(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['x_d'] - res_FB[idx][1]['x_a']) - \
                    np.cos(res_FB[idx][1]['psi_d']) * (res_FB[idx][1]['y_d'] - res_FB[idx][1]['y_a'])
    y_error_FF = np.sin(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['x_d'] - res_FF[idx][1]['x_a']) - \
                    np.cos(res_FF[idx][1]['psi_d']) * (res_FF[idx][1]['y_d'] - res_FF[idx][1]['y_a'])
    
    if path.name == 'left_turn_dominant_ditched_ellipsoid':
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_d'], 'k--', label='Desired')
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_a'], 'b', label='FB')
        ax[0,0].plot(res_FF[idx][1]['time'], res_FF[idx][1]['x_a'], 'r', label='FB + FF')
        ax[0,0].plot(res_FB[idx][1]['time'], res_FB[idx][1]['x_d'], 'k--', label='')
        ax[0,0].set_ylabel('X [m]')
        ax[0,0].grid()
        ax[0,0].set_ylim(-4, 9)
        ax[0,0].legend()

        ax[1,0].plot(res_FB[idx][1]['time'], x_error_FB, 'b', label='FB')
        ax[1,0].plot(res_FF[idx][1]['time'], x_error_FF, 'r', label='FB + FF')
        ax[1,0].grid()
        ax[1,0].set_xlabel('Time [s]')
        ax[1,0].set_ylabel(r'$\epsilon_x$ [m]')
        ax[1,0].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax[1,0].set_ylim(-0.55, 0.55)

        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_d'], 'k--', label='Desired')
        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_a'], 'b', label='FB')
        ax[0,1].plot(res_FF[idx][1]['time'], res_FF[idx][1]['y_a'], 'r', label='FB + FF')
        ax[0,1].plot(res_FB[idx][1]['time'], res_FB[idx][1]['y_d'], 'k--', label='')

        ax[0,1].grid() 
        ax[0,1].set_ylabel('Y [m]')
        ax[0,1].set_ylim(-4, 9)
        ax[0,1].legend()

        ax[1,1].plot(res_FB[idx][1]['time'], y_error_FB, 'b', label='FB')
        ax[1,1].plot(res_FF[idx][1]['time'], y_error_FF, 'r', label='FB + FF')
        ax[1,1].grid()
        ax[1,1].set_xlabel('Time [s]')
        ax[1,1].set_ylabel(r'$\epsilon_y$ [m]')
        ax[1,1].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax[1,1].set_ylim(-0.55, 0.55)

        # fig.suptitle(client_idx[idx] +': Left Turn Dominant Ditched Ellipsoid')

ax[0,0].label_outer()
for ax in ax.flat:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

if SAVE:
    fig.savefig('img/plots_ctrl_performance/tracking_error_client_IV.pdf')


print(f"Evaluate the FF model on all test paths!")
# 2x2 plot for each tets path
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# loop over ax and plot a sinues in each plot
for (path_idx, pp), ax in zip(enumerate(test_paths), axs.flat):
    _, log_traj_0 = eval_FF_model(FF_model=None, FF_type=None, FF_input=None, path=pp, cfg=config)
    _, log_traj_analytic = eval_FF_model(FF_model=None, FF_type='analytic', FF_input=None, path=pp, cfg=config)

    ax.plot(log_traj_0['x_a'], log_traj_0['y_a'], 'r', label='FB')
    ax.plot(log_traj_analytic['x_a'], log_traj_analytic['y_a'], 'm', label='FB + FF (analytic)')
    ax.plot(pp.x_fine, pp.y_fine, 'k--', label='Desired')

    ax.set_title(client_idx[path_idx] + ': ' + pp.name)
    ax.grid()
    ax.legend()
    plt.pause(0.25)

plt.tight_layout()
if SAVE:
    fig.savefig('img/plots_ctrl_performance/paths_test_FF.pdf')

print("Done.")
