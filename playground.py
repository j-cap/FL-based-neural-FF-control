"""

Script to test dynamic models and controllers in a closed-loop simulation

19.12.2024:
Testing the DynBicycleModel and the TrajectoryController classes
doesn't work as expected -> more work needs to be done there

"""

from car_models import DynBicycleModel
from paths import create_spline_paths
import json
import matplotlib.pyplot as plt
import numpy as np
from utils_FL import ClientNode
import os
import pandas as pd 




with open("config.json") as f:
    config = json.load(f)[0]

if os.getcwd().split('\\')[-1] == 'paper_FedFF':
    path_file = 'paths.json'

PATHS = create_spline_paths(file=path_file, real_world=config['simulation']['real_world'])
train_paths = PATHS[3]

path = train_paths
init_state = {'x': path.x_fine[0], 'y': path.y_fine[0], 'psi': path.psi_fine[0], 
              'dpsi': 0, 'v_x': path.v_long_profile[0], 'v_y': 0}

car = DynBicycleModel(initial_state=init_state, cfg=config)

traj = path.get_trajectory(config['simulation']['end_time'], config['simulation']['timestep'])
traj['vl'] = np.ones_like(traj['vl']) * traj['vl'][0]

time = traj['time']

log_traj_data = {'time': [], 'x_d': [], 'y_d': [], 'psi_d': [], 
                    'vl': [],   'x_a': [], 'y_a': [], 'psi_a': [], 
                    'des_velocity': [], 'u_steer': [], 'u_speed': [], 
                    'u_steer_FB': [], 'u_steer_FF': [],
                    'dpsi_actual': [], 'dpsi_desired': [], 
                    'des_curvature': [], 'actual_curvature': []}
# Simulation
dpsi_actual = 0
dt = config['simulation']['timestep']
for i, t in enumerate(time):

    x_a, y_a, psi_a, dpsi_a, v_x_a, v_y_a = car.state
    v_a = car.v_x
    x_d, y_d, psi_d = traj['x'][i], traj['y'][i], traj['psi'][i]

    desired_curvature = traj['kappa'][i]
    desired_velocity = traj['vl'][i]        
    dpsi_desired = desired_velocity * desired_curvature
    actual_curvature = dpsi_actual / desired_velocity

    # if FF_type == 'analytic':
    #     u_steer_FF = FF_yawrate_ctrl(desired_curvature, car_model.L) * car_model.angle2input
    # elif FF_type == 'model':
    #     if FF_input == 'desired':
    #         X_input = torch.tensor([desired_curvature, desired_velocity], dtype=torch.float32)
    #     elif FF_input == 'actual':
    #         X_input = torch.tensor([actual_curvature, desired_velocity], dtype=torch.float32)
    #     else:
    #         raise ValueError('FF_input must be either "desired" or "actual"')
    #     y_out = FF_model(X_input).detach().numpy()
    #     u_steer_FF = y_out[0]
    # else:
    u_steer_FF = 0

    # u_steer_FB = FB_yawrate_ctrl(dpsi_desired, dpsi_actual, controller, t) * car_model.angle2input
    u_steer_FB, u_speed_FB = car.traj_ctrl.control(x_a=x_a, x_d=x_d, 
                                                   y_a=y_a, y_d=y_d, 
                                                   psi_a=psi_a, psi_d=psi_d, 
                                                   dpsi_a=dpsi_actual, dpsi_d=dpsi_desired, 
                                                   v_a= v_a, v_d=traj['vl'][i], v_0=traj['vl'][0])

    # scale u1 & u2 to the input range of the car
    u1 = u_steer_FF + u_steer_FB 
    u1 = car.steering_rate_ctrl.control(u1)
    # u_speed_FB = 0
    u2 = (desired_velocity + u_speed_FB) * car.speed2input

    next_state = car.update(np.array([u1, u2]), dt)
    dpsi_actual = next_state[3]

    log_traj_data['time'].append(t)
    log_traj_data['x_a'].append(car.state[0])
    log_traj_data['y_a'].append(car.state[1])
    log_traj_data['psi_a'].append(car.state[2])
    log_traj_data['x_d'].append(x_d)
    log_traj_data['y_d'].append(y_d)
    log_traj_data['psi_d'].append(psi_d)
    log_traj_data['vl'].append(car.state[4])
    log_traj_data['des_velocity'].append(desired_velocity)
    log_traj_data['u_steer'].append(u1)
    log_traj_data['u_speed'].append(u2)
    log_traj_data['dpsi_actual'].append(dpsi_actual)
    log_traj_data['dpsi_desired'].append(dpsi_desired)
    log_traj_data['u_steer_FB'].append(u_steer_FB)
    log_traj_data['u_steer_FF'].append(u_steer_FF)
    log_traj_data['des_curvature'].append(desired_curvature)
    log_traj_data['actual_curvature'].append(actual_curvature)

df = pd.DataFrame(log_traj_data)



print(f"Shape of df: {df.shape}")

plt.figure()
plt.plot(df['time'], df['x_a'], 'r', label='x_a')
plt.plot(df['time'], df['x_d'], 'k--', label='x_d')

plt.plot(df['time'], df['y_a'], 'b', label='y_a')
plt.plot(df['time'], df['y_d'], 'k--', label='y_d')
plt.grid()
plt.show()

plt.figure()
plt.plot(df['des_velocity'], 'k--', label='desired velo')
plt.plot(df['vl'], 'r', label='actual velo')
plt.legend()
plt.ylim([-50,50])

plt.figure()
# plt.plot(df['time'], df['u_steer_FB'], 'r', label='u_steer_FB')
plt.plot(df['time'], df['u_steer'], 'b', label='u_steer')
plt.plot(df['time'], df['u_steer_FF'], 'm', label='u_steer_FF')
plt.plot(df['time'], df['u_steer_FB'], 'g', label='u_steer_FB')
plt.plot(df['time'], df['u_speed'], 'r', label='u_speed')
plt.legend()
plt.ylim([-100, 100])

print('done')
