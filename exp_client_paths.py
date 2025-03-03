"""

Create the plots for the client paths.

Date: 2025-01-10
Author: WeberJ 

"""


from paths import create_spline_paths
import json
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

SAVE = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
client_idx = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

with open("config.json") as f:
    config = json.load(f)[0]

RNG_SEED = config['simulation']['random_seed']

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
test_path_names = [path.name for path in test_paths]

print(f"Train path indices: {train_path_idx}")
print(f"Test path indices: {test_path_idx}")

v_long_max = max([path.v_long_profile.max() for path in PATHS])
v_long_min = min([path.v_long_profile.min() for path in PATHS])

# Define custom colormap
colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue, Green, Yellow, Red
nodes = [0.0, 0.5, 0.75, 1.0]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))


def plot_6_paths(paths):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for ax, path, c_idx in zip(axs.flatten(), paths, client_idx):
        T_end = np.trapz(1 / path.v_long_profile, dx=(path.t_fine[1] - path.t_fine[0])) * path.length
        points = np.array([path.x_fine, path.y_fine]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(v_long_min, v_long_max)
        lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=2)
        lc.set_array(path.v_long_profile)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        if path.name in test_path_names:
            col = 'red'
        else:
            col = 'blue'
        ax.set_title(c_idx + ': ' + path.name + f" - T_end = {T_end.round():.0f}s", color=col)
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2)

        ax.grid()
        ax.set_xlim([-12, 12])
        ax.set_ylim([-12, 12])
        cbar = fig.colorbar(lc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Desired Velocity [m/s]')
        cbar.set_ticks(np.linspace(v_long_min, v_long_max, num=5))
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

    plt.tight_layout()

    return fig

fig1_6 = plot_6_paths(PATHS[:6])
fig7_12 = plot_6_paths(PATHS[6:])

if SAVE:
    fig1_6.savefig('img/plots_clients/paths_1_6.pdf')
    fig7_12.savefig('img/plots_clients/paths_7_12.pdf')

fig = plt.figure(figsize=(10, 10))
for path in PATHS:
    points = np.array([path.x_fine, path.y_fine]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(v_long_min, v_long_max)
    lc = LineCollection(segments, cmap=custom_cmap, norm=norm)
    lc.set_array(path.v_long_profile)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
plt.grid()
plt.xlim([-12, 12])
plt.ylim([-12, 12])
cbar = plt.colorbar(lc, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Desired Velocity [m/s]')
cbar.set_ticks(np.linspace(v_long_min, v_long_max, num=5))
plt.show()

fig = plt.figure(figsize=(10, 10))
for path in PATHS:
    # plt.scatter(path.v_long_profile, path.curvature, s=2, marker='.', label=path.name)
    if path.name in test_path_names:
        col = 'red'
    else:
        col = 'blue'
    plt.plot(path.v_long_profile, path.curvature, c=col) # , label=path.name)
plt.xlabel('Desired Velocity [m/s]')
plt.ylabel('Desired Curvature [1/m]')
plt.grid()
plt.legend(['Test paths', 'Train paths'])
plt.show()

if SAVE:
    fig.savefig('img/plots_clients/curvature_vs_velocity.pdf')


# get more infos on the paths
D = []
for i, path in enumerate(PATHS):
    d = {}
    d['name'] = path.name
    d['length'] = path.length
    d['absolute_curvature_max'] = np.abs(path.curvature).max()
    d['v_long_max'] = path.v_long_profile.max()
    d['v_long_min'] = path.v_long_profile.min()
    d['time_to_complete'] = np.trapz(1 / path.v_long_profile, dx=(path.t_fine[1] - path.t_fine[0])) * path.length
    D.append(d)
    # print(f"Path: {path.name}")
    # print(f"Length: {path.length:.2f} m")
    # print(f"Curvature: {path.curvature.max():.2f} 1/m")
    # print(f"Desired velocity: {path.v_long_profile.max():.2f} m/s")
    # print(f"Time to complete: {np.trapz(1 / path.v_long_profile, dx=(path.t_fine[1] - path.t_fine[0])) * path.length:.2f} s")
    # print('')

df = pd.DataFrame(D, index=range(1,13))
if SAVE:
    df.to_csv('results/clients/paths_info.csv')



# def plot_12_paths(paths):
fig3, axs = plt.subplots(4, 3, figsize=(14, 10))
counter = 0
for ax, path, c_idx in zip(axs.flatten(), PATHS, client_idx):
    T_end = np.trapz(1 / path.v_long_profile, dx=(path.t_fine[1] - path.t_fine[0])) * path.length
    points = np.array([path.x_fine, path.y_fine]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(v_long_min, v_long_max)
    lc = LineCollection(segments, cmap=custom_cmap, norm=norm, linewidth=2)
    lc.set_array(path.v_long_profile)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    if path.name in test_path_names:
        col = 'red'
    else:
        col = 'blue'
    ax.set_title(c_idx + ': ' + path.name) # + f" - T_end = {T_end.round():.0f}s", color=col)
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2)

    ax.grid()
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])

    if counter in [2, 5, 8, 11]:
        cbar = fig.colorbar(lc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Desired Velocity [m/s]')
        cbar.set_ticks(np.linspace(v_long_min, v_long_max, num=5))
    if counter in [0, 3, 6, 9]:
        ax.set_ylabel('Y [m]')
    if counter in [9, 10, 11]:
        ax.set_xlabel('X [m]')

    counter += 1

plt.tight_layout()
if SAVE:
    fig3.savefig('img/plots_clients/paths_1_12.pdf')

    # return fig

print(f'Done!')

