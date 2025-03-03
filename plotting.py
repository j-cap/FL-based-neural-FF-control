import pandas as pd # type: ignore 
import matplotlib.pyplot as plt
import numpy as np 
import os


# list all .csv files in /logs/ directory
def list_files():
    files = os.listdir('logs/')
    return [f for f in files if f.endswith('.csv')]


def plot_single(idx=0):
    logs = list_files()

    # load a .csv file from /logs/ directory
    df = pd.read_csv('logs/'+logs[idx])

    # plot the data
    plt.figure()
    plt.plot(df['time'], df['u_steer'], label='u_steer')
    plt.plot(df['time'], df['des_curvature'], label='des_curvature')
    plt.plot(df['time'], df['des_velocity'], label='des_velocity')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_all():
    logs = list_files()

    max_abs_kappa = 0
    max_vl = 0
    # 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for log in logs:
        log_FF = pd.read_csv('logs/'+log)
        p = ax.scatter(log_FF['des_velocity'], log_FF['des_curvature'], log_FF['u_steer'], label=log)
        if log_FF['des_curvature'].abs().max() > max_abs_kappa:
            max_abs_kappa = log_FF['des_curvature'].abs().max()
        if log_FF['des_velocity'].max() > max_vl:
            max_vl = log_FF['des_velocity'].max()
    plt.xlabel('Desired Velocity')
    plt.ylabel('Desired Curvature')
    plt.xlim([0, max_vl])
    plt.ylim(np.array([-1,1])*max_abs_kappa)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_tracking_error_and_path(RMSEs, test_paths, test_path_idx, 
                                 title='Mean Tracking error by Path and Epoch: Federated Learning'):
    plt.rcParams.update({'font.size': 16})  # You can change 12 to any desired font size

    # bar chart of the RMSEs with one bar per path and grouped by epoch
    values = np.array(list(RMSEs.values()))
    n_bars = len(test_paths)
    x = np.arange(n_bars)
    bar_width = 0.5 / values.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.bar(x, values[0,:], width=bar_width, label='PI only')
    for i in range(1,values.shape[0]-1):
        ax.bar(x + i * bar_width, values[i,:], width=bar_width, label=f'Round {i}')
    ax.bar(x + (values.shape[0]-1) * bar_width, values[-1,:], width=bar_width, label='Analytic')
    ax.set_ylabel('Mean trajectory error [m]')
    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(['P'+str(i+1) for i in test_path_idx])
    ax.legend(title='')
    ax.set_yscale('log')
    ax.set_ylim([1e-2,1e-0])
    ax.set_yscale('linear')
    ax.set_ylim([1e-2, values.max()*1.2])
    plt.grid()
    plt.title(title)
    ax2 = fig.add_subplot(212)
    # add x-y plot for each path
    for i in range(len(test_paths)):
        ax2.plot(test_paths[i].x_fine, test_paths[i].y_fine, label=f"P{test_path_idx[i]+1}")
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim([-12, 12])
    ax2.set_ylim([-12, 12])
    ax2.axis('equal')
    plt.title('Test paths')
    plt.tight_layout()
    plt.show() 

    return


# plot_all()