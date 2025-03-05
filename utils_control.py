

import pandas as pd # type: ignore
import numpy as np
import torch 

def simulate_closed_loop_traj_follow(traj, car_model, FF_type=None, FF_model=None, FF_input=None, dt=0.05):
    """ Take the trajectory and perform the closed-loop simulation of the vehicle model. 
    
    Args:
    traj: dict, trajectory data
    car_model: BicycleModel, vehicle model
    controller: PID, controller
    FF_type: str, type of FF model {'model', 'analytic', None}
    FF_model: torch model, FF model - only necessary if FF_type='model'
    FF_inptu: str, input for the FF model {'desired', 'actual'}
    dt: float, time step

    Returns:
    df: pd.DataFrame, data from the FF model

    """
    time = traj['time']

    log_traj_data = {'time': [], 'x_d': [], 'y_d': [], 'psi_d': [], 
                     'vl': [],   'x_a': [], 'y_a': [], 'psi_a': [], 
                     'des_velocity': [], 'u_steer': [], 'u_speed': [], 
                     'u_steer_FB': [], 'u_steer_FF': [],
                     'dpsi_actual': [], 'dpsi_desired': [], 
                     'des_curvature': [], 'actual_curvature': []}
    # Simulation
    dpsi_actual = 0
    if FF_type == 'model':
        FF_model.eval()    
    for i, t in enumerate(time):

        x_a, y_a, psi_a = car_model.state
        v_a = car_model.vl
        x_d, y_d, psi_d = traj['x'][i], traj['y'][i], traj['psi'][i]

        desired_curvature = traj['kappa'][i]
        desired_velocity = traj['vl'][i]        
        dpsi_desired = desired_velocity * desired_curvature
        actual_curvature = dpsi_actual / desired_velocity

        if FF_type == 'analytic':
            u_steer_FF = FF_yawrate_ctrl(desired_curvature, car_model.L) # * car_model.angle2input
        elif FF_type == 'model':
            if FF_input == 'desired':
                X_input = torch.tensor([desired_curvature, desired_velocity], dtype=torch.float32)
            elif FF_input == 'actual':
                X_input = torch.tensor([actual_curvature, desired_velocity], dtype=torch.float32)
            else:
                raise ValueError('FF_input must be either "desired" or "actual"')
            y_out = FF_model(X_input).detach().numpy()
            u_steer_FF = y_out[0]
        else:
            u_steer_FF = 0

        u_steer_FB, u_speed_FB = car_model.traj_ctrl.control(x_a=x_a, x_d=x_d, 
                                                             y_a=y_a, y_d=y_d, 
                                                             psi_a=psi_a, psi_d=psi_d, 
                                                             dpsi_a=dpsi_actual, dpsi_d=dpsi_desired, 
                                                             v_a=v_a, v_d=traj['vl'][i], v_0=traj['vl'][0])

        # scale u1 & u2 to the input range of the car
        u1 = u_steer_FF + u_steer_FB 
        u1 = car_model.steering_rate_ctrl.control(u1)
        u2 = (desired_velocity + u_speed_FB) * car_model.speed2input

        dpsi_actual = car_model.update([u1, u2], dt)

        log_traj_data['time'].append(t)
        log_traj_data['x_a'].append(car_model.state[0])
        log_traj_data['y_a'].append(car_model.state[1])
        log_traj_data['psi_a'].append(car_model.state[2])
        log_traj_data['x_d'].append(x_d)
        log_traj_data['y_d'].append(y_d)
        log_traj_data['psi_d'].append(psi_d)
        log_traj_data['vl'].append(car_model.vl)
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

    return df


def FF_yawrate_ctrl(desired_curvature: float, L: float) -> float:
    """ Analytic feedforward control for the steering angle. """
    delta = np.arctan(L * desired_curvature)
    return delta
