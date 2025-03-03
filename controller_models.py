"""

Some controller implementations

"""

import numpy as np
from paths import SplinePath
import matplotlib.pyplot as plt
import json 

from utils_control import simulate_closed_loop
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
    
class FFmodelSimple(nn.Module):
    def __init__(self, n_neurons=5):
        self.n_neurons = n_neurons
        super(FFmodelSimple, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(2, n_neurons))
        self.fc2 = spectral_norm(nn.Linear(n_neurons, n_neurons))
        self.fc3 = spectral_norm(nn.Linear(n_neurons, 1))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _create_copy(self): 
        """ Create a fresh copy of the model. """
        model_copy = FFmodelSimple(n_neurons=self.n_neurons)
        model_copy.load_state_dict(self.state_dict(), strict=True)
        return model_copy

    def _plot(self):

        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.forward(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)).item()
        plt.contourf(X, Y, Z, levels=100)
        plt.colorbar()
        plt.show()
        return

# create a dataset class
class FFmodelData(Dataset):
    def __init__(self, df, FF_input='desired'):
        self.df = df
        if FF_input == 'actual':
            self.X = df[['actual_curvature', 'vl']].values
        elif FF_input == 'desired':
            self.X = df[['des_curvature', 'des_velocity']].values
        else:
            raise ValueError('FF_input must be either "desired" or "actual"')
        self.y = df['u_steer'].values

    def __describe__(self):
        print(f"Dataset shape: {self.X.shape}")
        print(f"X / y - {self.df.columns}")
        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def train_FF_model(model: nn.Module, 
                   optimizer: nn.Module, 
                   trainloader: DataLoader, 
                   loss_fn: nn.Module, 
                   epochs: int=5):
    """ Train NN model 
    Args:
        model: torch model
        optimizer: torch optimizer
        trainloader: torch DataLoader
        loss_fn: torch loss function
        epochs: int, number of epochs to train the model    
    """
    train_losses = []
    model.train()
    for epoch in range(epochs):
        train_loss = []
        for i, (X, y) in enumerate(trainloader):
            optimizer.zero_grad()
            y_pred = model(X).flatten()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_losses.append(np.mean(train_loss))
        print(f"\t\tEpoch: {epoch} | Loss: {train_losses[epoch]**0.5:.6f}") # loss is MSE, print RMSE
    
    return 

class PD:
    def __init__(self, Kp, Kd):
        self.Kp = Kp
        self.Kd = Kd
        self.prev_error = 0
        self.t_prev = -1e-8

    def control(self, error, t):
        dt = t - self.t_prev
        self.t_prev = t
        derivative = (error - self.prev_error)
        self.prev_error = error
        return self.Kp * error + self.Kd * derivative
   

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.t_prev = -1e-8
        self.integral_max = 1.75

    def control(self, error, t):
        dt = t - self.t_prev
        self.integral += error * dt
        derivative = (error - self.prev_error) * dt
        self.prev_error = error
        self.t_prev = t

        if self.integral > self.integral_max:
            # print("Integral too large")
            self.integral = self.integral_max * np.sign(self.integral)

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    

class TrajectoryFollower:
    """ 
    Control law from: 
    Althoff, Matthias, and John M. Dolan. 
        "Online verification of automated road vehicles using reachability analysis." IEEE Transactions on Robotics 30.4 (2014): 903-918.
    """
    def __init__(self, gains: list):
        # v0 = path.v_long_profile[0]
        # kGain = v0 / (max(abs(path.v_long_profile)) + 1e-4)
        self.K1 = gains[0] # lateral error
        self.K2 = gains[1] # orientation error
        self.K3 = gains[2] # yaw rate error
        self.K4 = gains[3] # steering velocity
        self.K5 = gains[4] # longitudinal error 
        self.K6 = gains[5] # velocity error -> skip speed control

    def control(self, 
                x_a, x_d, 
                y_a, y_d,
                psi_a, psi_d, 
                dpsi_a, dpsi_d, 
                v_a: float, v_d: float=1, 
                v_0: float = 0):
        
        
        e_x = x_d - x_a
        e_y = y_d - y_a
        e_psi = psi_d - psi_a
        # wrap e_psi to interval [-pi, pi]
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        e_dpsi = dpsi_d - dpsi_a

        eps_x = np.cos(psi_d) * e_x + np.sin(psi_d) * e_y
        eps_y = -np.sin(psi_d) * e_x + np.cos(psi_d) * e_y

        kGain = v_0 / (np.abs(v_d) + 1e-4)
        kGain = 0.05
        u_steer_FB = kGain * (self.K1 * eps_y + self.K2 * e_psi + self.K3 * e_dpsi)
        u_speed_FB = self.K5 * eps_x # + self.K6 * (v_d - v_a) # -> skip speed control
    
        return u_steer_FB, u_speed_FB
    
class TrajFollower_NonlinearBM(TrajectoryFollower):
    def __init__(self, car_model, gains: list):
        super().__init__(gains, )
        self.car = car_model

    def control(self, 
                x_a, x_d, 
                y_a, y_d,
                psi_a, psi_d, 
                dpsi_a, dpsi_d, 
                v_a: float, v_d: float=1, 
                v_0: float = 0):
        
        kGain = v_0 / (np.abs(v_d) + 1e-4)
        # calculate errors
        e_x = x_d - x_a
        e_y = y_d - y_a
        e_psi = psi_d - psi_a
        # wrap e_psi to interval [-pi, pi]
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        e_dpsi = dpsi_d - dpsi_a

        eps_x = np.cos(psi_d) * e_x + np.sin(psi_d) * e_y
        eps_y = -np.sin(psi_d) * e_x + np.cos(psi_d) * e_y

        # calcuate steering FB
        u_steer_FB = kGain * (self.K1 * eps_y + self.K2 * e_psi + self.K3 * e_dpsi)
        u_steer_FB = np.clip(u_steer_FB, -self.car.max_steer_angle, self.car.max_steer_angle)

        # calculate torque FB
        acc_input = self.K5 * eps_x + self.K6 * (v_d - v_a)
        torque_sum = self.car.mass * acc_input * self.car.r_dyn
        if torque_sum > 0:
            u_torque_FB = np.array([0.5, 0.5]) * torque_sum
        else:
            u_torque_FB = np.array([0.5, 0.5]) * torque_sum
        
        return u_steer_FB, u_torque_FB
    
class SteeringRateController:
    def __init__(self, 
                 car_model,
                 gains: list):
                  
        self.Kp = gains[0]
        self.u_steer_prev = None
        self.dt = car_model.dt
        self.car = car_model

    def control(self, u_steer_d):

        if self.u_steer_prev is None:
            self.u_steer_prev = u_steer_d

        if np.abs(u_steer_d) > self.car.max_steer_angle:
            u_steer_d = self.car.max_steer_angle * np.sign(u_steer_d)

        u_steering_rate = self.Kp * (u_steer_d - self.u_steer_prev)
        if np.abs(u_steering_rate) > self.car.max_steer_rate:
            u_steering_rate = self.car.max_steer_rate * np.sign(u_steering_rate)
        # euler forward 
        u_steer = self.u_steer_prev + u_steering_rate * self.dt
        
        np.clip(u_steer, -self.car.max_steer_angle, self.car.max_steer_angle)
        self.u_steer_prev = u_steer

        return u_steer