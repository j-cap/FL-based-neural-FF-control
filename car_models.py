"""
File contains the car models used in the experiments. 

Author: Jakob Weber
Date: 21.03.2024


Contains the following dynamic models of the car:
    - Point mass
    - Bicycle model
    - Linearized bicycle model
    - Linearized bicycle model with sideslip
    - Linearized bicycle model with sideslip and yaw rate

"""

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from scipy import signal # type: ignore
import torch

from controller_models import TrajectoryFollower, SteeringRateController, TrajFollower_NonlinearBM

class PointMass:
    """ Describes a point mass in 2D. """
    def __init__(self, 
                 m: float, 
                 xpos: float,
                 ypos: float):
        self.m = m
        self.xpos = xpos
        self.ypos = ypos
        self.v_x = 0
        self.v_y = 0

    def update(self, u: np.ndarray, dt: float) -> list:
        """ Euler integration of the point mass. 
        
        Parameters:
        u: np.ndarray - The control input.
        dt : float - The time step.

        returns: 
        list - The new state of the point mass.
        """
        self.v_x += u[0] * dt
        self.v_y += u[1] * dt
        self.xpos += self.v_x * dt
        self.ypos += self.v_y * dt
        return [self.xpos, self.ypos, self.v_x, self.v_y] 
    
class BicycleModel:
    """ Describes a bicycle model in 2D. """
    def __init__(self, 
                 initial_state: dict, 
                 cfg: dict):
        """
        Describes a bicycle model in 2D.
        """
        np.random.seed(cfg['simulation']['random_seed'])
        torch.manual_seed(cfg['simulation']['random_seed'])

        self.L = cfg['car']['kinematic_bm']['length']
        self.mass = cfg['car']['kinematic_bm']['mass']
        self.kSpeed = cfg['car']['general']['kSpeed']
        self.max_steer_angle = np.deg2rad(cfg['car']['general']['max_steering_angle_degree'])
        self.max_steer_rate = np.deg2rad(cfg['car']['general']['max_steering_rate_degree'])
        self.speed2input = 1 / self.kSpeed
        self.steer2angle = np.deg2rad(cfg['car']['general']['max_steering_angle_degree'])
        self.angle2input = 1/self.steer2angle

        self.xpos = initial_state["x"]
        self.ypos = initial_state["y"]
        self.psi = initial_state["psi"]
        self.vl = initial_state["vl"]

        self.dt = cfg['simulation']['timestep']
        self.noise_level = cfg['simulation']['noise_level']

        # according to state-variable filter in slx
        SS = signal.StateSpace([cfg['car']['general']['speed_FB_pole']], [1], [-self.kSpeed * cfg['car']['general']['speed_FB_pole']], [0])
        self.discrete_speed_filter = {'filter': signal.cont2discrete((SS.A, SS.B, SS.C, SS.D), dt=self.dt, method='zoh'),
                                        'state': -self.vl /(self.kSpeed*cfg['car']['general']['speed_FB_pole'])}

        # transform feedback pole to time constant: s = -1 / T --> T = -1 / s
        time_constant = -1 / cfg['car']['general']['speed_FB_pole']
        self.PT1 = PT1_filter(K=1, T=time_constant, Ts=self.dt, init_state = self.vl)
        self.state = np.array([self.xpos, self.ypos, self.psi])

        # initialize controller
        self.traj_ctrl = TrajectoryFollower(gains=cfg['controller']['trajectory']['gains'])
        self.steering_rate_ctrl = SteeringRateController(gains=[cfg['controller']['steering_rate']['Kp']], car_model=self)

        return

    def update(self, u_: np.ndarray, dt: float) -> float:
        """ Euler integration of the bicycle model. 
        
        Parameters:
        u: np.ndarray - The control input [steering angle, acceleration].
        dt : float - The time step.

        returns: 
        list - The new state of the bicycle model.
        """
        # clip the input
        u = self._clip_input(u_)
        # convert the steering input to the steering angle of the model
        delta = u[0] # * self.steer2angle
        # filter the speed input
        # vl = self._filter(u[1] * self.kSpeed)
        vl = self._pt1_filter(u[1] * self.kSpeed)
        x, y, psi = self.state
        
        # dynamics
        dx = vl * np.cos(psi)
        dy = vl * np.sin(psi)
        dpsi = vl / self.L * np.tan(delta)

        # Euler integration
        x += dx * dt + np.random.normal(0, self.noise_level['position']) 
        y += dy * dt + np.random.normal(0, self.noise_level['position'])
        psi += dpsi * dt + np.random.normal(0, self.noise_level['orientation'])

        self.state = np.array([x, y, psi])
        self.vl = vl

        return dpsi

    def _filter(self, u):
        A, B, C, D, dt = self.discrete_speed_filter['filter']
        x = self.discrete_speed_filter['state']
        dx = A.squeeze() * x + B.squeeze() * u
        y = C.squeeze() * x +  D.squeeze() * u
        self.discrete_speed_filter['state'] += dx * dt
        return y
    
    def _pt1_filter(self, u):
        return self.PT1.step(u)

    def _clip_input(self, u):
        u1 = np.clip(u[0], -1, 1) # steering angle
        u2 = np.clip(u[1], -1, 1)  # desired speed
        return u1, u2

class DynBicycleModel:
    """ Dynamic bicycle model with nonlinear tire forces. """
    def __init__(self,
                 initial_state: dict,
                 cfg: dict):
        
        self.g = 9.81  # Gravitational acceleration

        # car parameters
        self.l_f = cfg['car']['mercedes_C220']["chassis"]['l_f']
        self.l_r = cfg['car']['mercedes_C220']["chassis"]['l_r']
        self.l = self.l_f + self.l_r
        self.m_c = cfg['car']['mercedes_C220']["chassis"]['mass']
        self.J_y = cfg['car']['mercedes_C220']["chassis"]['J_y']
        self.rho_air = cfg['car']['mercedes_C220']["chassis"]['rho_air']
        self.mue = cfg['car']['mercedes_C220']['street_friction']

        self.kSpeed = cfg['car']['general']['kSpeed']
        self.speed2input = 1 / self.kSpeed
        self.max_steer_angle = np.deg2rad(cfg['car']['general']['max_steering_angle_degree'])
        self.max_steer_rate = np.deg2rad(cfg['car']['general']['max_steering_rate_degree'])
        self.steer2angle = np.deg2rad(self.max_steer_angle)
        self.angle2input = 1/self.steer2angle

        # magic formula parameters
        self.By = cfg['car']['mercedes_C220']['tire']['magic_y']['B']
        self.Cy = cfg['car']['mercedes_C220']['tire']['magic_y']['C']
        self.Dy = cfg['car']['mercedes_C220']['tire']['magic_y']['D']
        self.Ey = cfg['car']['mercedes_C220']['tire']['magic_y']['E']

        # initial state
        self.xpos = initial_state["x"]
        self.ypos = initial_state["y"]
        self.psi = initial_state["psi"]
        self.dpsi = initial_state["dpsi"]
        self.v_x = initial_state["v_x"] # longitudinal velocity
        self.v_y = initial_state["v_y"] # lateral velocity
        self.state = np.array([self.xpos, self.ypos, self.psi, self.dpsi, self.v_x, self.v_y])

        # simulation parameters 
        self.dt = cfg['simulation']['timestep']
        self.noise_level = cfg['simulation']['noise_level']

        # initialize controller
        self.traj_ctrl = TrajectoryFollower(gains=cfg['controller']['trajectory']['gains'])
        self.steering_rate_ctrl = SteeringRateController(gains=[cfg['controller']['steering_rate']['Kp']], car_model=self)

        # transform feedback pole for PT1 speed following to time constant: s = -1 / T --> T = -1 / s
        time_constant = -1 / cfg['car']['general']['speed_FB_pole']
        time_constant = -1 / -10
        self.PT1 = PT1_filter(K=1, T=time_constant, Ts=self.dt, init_state = self.v_x)

        return
    
    def update(self, u_: np.ndarray, dt: float) -> np.ndarray:
        """ Euler integration of the dynamic bicycle model. 
        
        Parameters:
        u: np.ndarray - The control input [steering angle, long. velocity].
        dt : float - The time step.

        returns: 
        list - The new state of the dynamic bicycle model.
        """
        x, y, psi, dpsi, v_x, v_y = self.state
        # clip the input
        u = self._clip_input(u_)
        # convert the steering input to the steering angle of the model
        delta = u[0] * self.steer2angle
        # filter the speed input -> assume only forward wheel drive 
        v_x = self._pt1_filter(u[1] * self.kSpeed)

        # static load distribution
        FF_axle_z = self.m_c * self.g * np.array([self.l_r, self.l_f]) / (self.l)

        # swim angles
        alpha_f = np.arctan((v_y + self.l_f * dpsi) / abs(v_x + 1e-3)) - delta
        alpha_r = np.arctan((v_y - self.l_r * dpsi) / abs(v_x + 1e-3))
        # handling for steered wheels if angle is above 90 degrees
        d = np.abs(alpha_f) - np.pi / 2 
        if d > 0:
            alpha_f = np.sign(alpha_f) * (np.pi / 2 - d)
        alpha = np.array([alpha_f, alpha_r])
        # Pacejka magic formula for tire forces 
        FF_tire_y = FF_axle_z * self.mue * self.Dy * np.sin(self.Cy * np.arctan(self.By * alpha - self.Ey * (self.By * alpha - np.arctan(self.By * alpha))))

        # forces on the axles
        FF_axle_y_max = FF_axle_z * self.mue * self.Dy
        FF_axle_y_W = -FF_tire_y

        # tire forces on the axles in wheel coordinate system
        F_axle_xf_W = 0 # long. 
        F_axle_yf_W = FF_axle_y_W[0] # lateral, front wheel
        F_axle_yr_W = FF_axle_y_W[1] # lateral, rear wheel

        # dynamics
        a_q = 1 / self.m_c * (F_axle_yf_W * np.cos(delta) + F_axle_yr_W + F_axle_xf_W * np.sin(delta) - self.m_c * v_y * dpsi) # lateral acceleration, according to slides Introduction to vehicle dynamics, p. 17
        M_psi = 0 # additional jaw momentum

        # derivatives in real world coordinates
        dx = v_x * np.cos(psi) - v_y * np.sin(psi)
        dy = v_x * np.sin(psi) + v_y * np.cos(psi)
        dv_y = a_q
        dv_x = 0 # assume no acceleration in longitudinal direction
        dpsi_p = 1/self.J_y * (M_psi + self.l_f * (F_axle_yf_W * np.cos(delta) + F_axle_xf_W * np.sin(delta)) - self.l_r * F_axle_yr_W)

        # Euler integration
        x += dx * dt
        y += dy * dt
        dpsi += dpsi_p * dt
        psi += dpsi * dt
        v_x += dv_x * dt
        v_y += dv_y * dt

        self.state = np.array([x, y, psi, dpsi, v_x, v_y])

        return self.state

    def _pt1_filter(self, u):
        return self.PT1.step(u)

    def _clip_input(self, u):
        u1 = np.clip(u[0], -1, 1) # steering angle
        u2 = np.clip(u[1], -40, 40)  # desired speed
        # u2 = u[1]
        return u1, u2


class PT1_filter():
    def __init__(self, K, T, Ts, init_state=0):
        """ Discrete PT1 filter. """
        self.K = K  # gain
        self.T = T  # time constant
        self.Ts = Ts # sampling time
        self.state = init_state # initial state

        # Discretize the PT1 filter
        self.alpha = self.T / (self.T + self.Ts)
        self.beta = self.K * self.Ts / (self.T + self.Ts)

    def step(self, u):
        y = self.alpha * self.state + self.beta * u
        self.state = y
        return y


class DynamicBicycleModel:
    def __init__(self, 
                 initial_state: dict, 
                 cfg: dict):
        # Vehicle and tire parameters
        self.g = 9.81  # Gravitational acceleration
        
        # Dimensions
        self.n = 6  # state space
        self.m = 3  # control space
        self.p = 6  # output space

        # Labels
        self.name = 'Dynamic Bicyle Model with nonlinear tire model'
        self.state_label  = ['X', 'Y', 'v_long', 'v_lat', 'psi', 'dpsi']
        self.input_label  = ['delta', 'tau_f', 'tau_r']
        self.output_label = ['X', 'Y', 'v_long', 'v_lat', 'psi', 'dpsi']
        
        # Units
        self.state_units  = ['[m]','[m]', '[m/s]', '[m/s]', '[rad]','[rad/s]']
        self.input_units  = ['[rad]', '[N]', '[N]']
        self.output_units = ['[m]','[m]', '[m/s]', '[m/s]', '[rad]','[rad/s]']
        
        self.state = np.array(
            [initial_state["x"], initial_state["y"], 
             initial_state["v_long"], initial_state['v_lat'],
             initial_state["psi"], initial_state["dpsi"]], dtype=np.float64)

        # Extract parameters from the provided parameter dictionary
        self.mue        = cfg["car"]["mercedes_C220"]['street_friction']
        self.chassis    = cfg["car"]["mercedes_C220"]['chassis']
        self.tire       = cfg["car"]["mercedes_C220"]['tire']
        
        # Chassis parameters
        self.l_f        = self.chassis['l_f']
        self.l_r        = self.chassis['l_r']
        self.l          = self.l_f + self.l_r
        self.m_c        = self.chassis['mass']
        self.rho_air    = self.chassis['rho_air']
        self.c_wA       = self.chassis['c_wA']
        self.J_y        = self.chassis['J_y']
        
        # Tire parameters
        self.m_t        = self.tire['mass']
        self.c_fr       = self.tire['c_fr']
        self.r_dyn      = self.tire['r_dyn']
        
        # Magic Formula Parameters
        self.By = self.tire['magic_y']['B']
        self.Cy = self.tire['magic_y']['C']
        self.Dy = self.tire['magic_y']['D']
        self.Ey = self.tire['magic_y']['E']
        
        # Calculated parameters
        self.mass = self.m_c + sum(self.m_t)

        self.dt = cfg['simulation']['timestep']
        self.noise_level = cfg['simulation']['noise_level']

        self.max_steer_angle = cfg['car']["general"]["max_steering_angle_degree"]
        self.max_steer_rate = np.deg2rad(cfg['car']["general"]["max_steering_rate_degree"])
        self.steer2angle = np.deg2rad(self.max_steer_angle)
        self.angle2input = 1/self.steer2angle

        # # initialize controller
        self.traj_ctrl = TrajFollower_NonlinearBM(gains=cfg['controller']['trajectory']['gains'], car_model=self)
        self.steering_rate_ctrl = SteeringRateController(gains=[cfg['controller']['steering_rate']['Kp']], car_model=self)

        
    def update(self, u):
        """
        State variables: 
            ['X', 'Y', 'v_long', 'v_lat', 'psi', 'dpsi']
        
        System inputs:
            ['steer_input', 'tau_f', 'tau_r']
        
        """
        # State variables
        x = self.state
        v_long = x[2]
        v_lat = x[3]
        psi = x[4]
        dpsi = x[5]
        cos_psi,  sin_psi = np.cos(psi), np.sin(psi)
        
        # System inputs
        delta = np.clip(u[0], -1, 1) * self.steer2angle # transform input to steering angle again
        torque = u[1:3]
        
        # ----- tires and wheels -----
        # Static load distribution
        FF_axle_z = self.g * (self.m_c * np.array([self.l_r, self.l_f]) / self.l + np.array([self.m_t[0] + self.m_t[1], self.m_t[2] + self.m_t[3]]))
        
        # Friction forces
        FF_axle_fr = FF_axle_z * self.c_fr * np.tanh(v_long / (self.r_dyn * 1e3))  # is this correct??? 
        
        # longitudinal tire forces (rotational wheel dynamics are neglected and these are the resulting forces)
        FF_axle_x_W = torque / self.r_dyn - FF_axle_fr
        
        # Slip angles
        alpha_f = np.arctan((v_lat + self.l_f * dpsi) / abs(v_long + 1e-3)) - delta
        alpha_r = np.arctan((v_lat - self.l_r * dpsi) / abs(v_long + 1e-3))
        alpha = np.array([alpha_f, alpha_r])
        
        # Nonlinear static tire forces in lateral direction
        FF_tire_y = FF_axle_z * self.mue * self.Dy * np.sin(self.Cy * np.arctan(self.By * alpha - self.Ey * (self.By * alpha - np.arctan(self.By * alpha))))
        FF_axle_y_W = -FF_tire_y
        
        # Limitation of forces according to the friction circle
        F_axle_shear_norm = (FF_axle_x_W**2 + FF_axle_y_W**2)**0.5 / (FF_axle_z * self.mue)
        i_limit = F_axle_shear_norm > 1
        F_axles_shear_norm_in_limit = np.sqrt(F_axle_shear_norm[i_limit])
        FF_axle_x_W[i_limit] /= F_axles_shear_norm_in_limit
        FF_axle_y_W[i_limit] /= F_axles_shear_norm_in_limit
        
        # Tire forces on the axes in the wheel-coordinate system
        F_axle_xf_W = FF_axle_x_W[0]
        F_axle_xr_W = FF_axle_x_W[1]
        F_axle_yf_W = FF_axle_y_W[0]
        F_axle_yr_W = FF_axle_y_W[1]
        
        # Environmental forces
        R_w_to_v = np.array([[cos_psi, sin_psi], 
                             [-sin_psi, cos_psi]])

        # ToDo: add air-velo and slope as disturbances
        print(f"Air velo and slope are not implemented yet!")
        air_velocity = [0,0]
        delta_v = np.array([v_long, v_lat]) - R_w_to_v @ air_velocity
        delta_v = np.array([0,0])
        F_air = -0.5 * self.rho_air * self.c_wA * delta_v * np.abs(delta_v)
        # theta_slope = 0
        # phi_slope = 0
        # F_slope_w = self.mass * self.g * np.array([np.sin(theta_slope), -np.sin(phi_slope)])
        F_slope_w = np.array([0, 0])
        F_slope = R_w_to_v @ F_slope_w
        FF_ext = F_slope + F_air
        
        # Accelerations in car frame
        a_long = 1 / self.mass * (FF_ext[0] + F_axle_xr_W + F_axle_xf_W * np.cos(delta) - F_axle_yf_W * np.sin(delta))
        a_lat  = 1 / self.mass * (FF_ext[1] + F_axle_yr_W + F_axle_xf_W * np.sin(delta) + F_axle_yf_W * np.cos(delta))
        # additional jaw momentum
        M_psi = 0
        
        # Derivatives
        xp = v_long * cos_psi - v_lat * sin_psi
        yp = v_long * sin_psi + v_lat * cos_psi
        v_long_p = a_long + v_lat * dpsi
        v_lat_p = a_lat - v_long * dpsi
        ddpsi = 1 / self.J_y * (M_psi + self.l_f * (F_axle_yf_W * np.cos(delta) + F_axle_xf_W * np.sin(delta)) - self.l_r * F_axle_yr_W)
        dpsi = dpsi + ddpsi * self.dt
        # Return the derivatives vector
        dx = np.array([xp, yp, v_long_p, v_lat_p, dpsi, ddpsi])        

        # Euler integration
        x_new = x + dx * self.dt
        self.state = x_new

        return dx
    
class LateralDynamicsBicycleWithSpeedInput:

    def __init__(self):

        # Dimensions
        self.n = 5  # state space
        self.m = 2  # control space
        self.p = 5  # output space

        # Labels
        self.name = 'Lateral Dynamic Bicyle Model'
        self.state_label = ['v_y','dtheta','theta','X','Y']
        self.input_label = ['delta', 'v_x']
        self.output_label = ['v_y','dtheta','theta','X','Y']
        
        # Units
        self.state_units = ['[m/s]','[rad/s]','[rad]','[m]','[m]']
        self.input_units = ['[rad]', '[m/sec]']
        self.output_units = ['[m/s]','[rad/s]','[rad]','[m]','[m]']      

        # Model parameters
        self.L_rear = 1
        self.L_front = 1
        self.L = self.L_rear + self.L_front
        self.width = 0.2
        self.mass = 1
        self.G = 9.81
        self.dt = 0.05

        self.Iz = 1/12 * self.mass * (self.lenght**2 + self.width**2)

    def update(self, u):
        """ Euler integration of the bicycle model. """
        F_yf, F_yr = self.tiremodel(x, u)

        v_y, dtheta, theta, x, y = self.state 

        # Dynamics
        v_y_dot = (F_yf * np.cos(u[0]) + F_yr) / self.mass - u[1] * dtheta
        dtheta_dot = (self.L_front * F_yf * np.cos(u[0]) - self.L_rear * F_yr) / self.Iz
        theta_dot = dtheta
        x_dot = u[1] * np.cos(theta) - v_y * np.sin(theta)
        y_dot = u[1] * np.sin(theta) + v_y * np.cos(theta)

        # Euler integration
        v_y += v_y_dot * self.dt
        dtheta += dtheta_dot * self.dt
        theta += theta_dot * self.dt
        x += x_dot * self.dt
        y += y_dot * self.dt

        self.dynamics = [v_y_dot, dtheta_dot, theta_dot, x_dot, y_dot]
        self.state = [v_y, dtheta, theta, x, y]




    def tiremodel(self, x, u):
        """ Piecewise tire model. """
        # Tire parameters
        C_alpha = 1
        C_beta = 1
        D_alpha = 1
        D_beta = 1

        # Slip angles
        alpha_f = np.arctan((x[1] + self.L_rear * x[2]) / x[0]) - u[0]
        alpha_r = np.arctan((x[1] - self.L_front * x[2]) / x[0])

        # Tire forces
        F_yf = D_alpha * np.sin(C_alpha * np.arctan(B_alpha * alpha_f))
        F_yr = D_beta * np.sin(C_beta * np.arctan(B_beta * alpha_r))

        return F_yf, F_yr
    
    def piecewise_tiremodel(self, x, u):

        # tire-road friction coefficient
        mu = 0.9
        # Compute normal forces on tires
        F_nf = self.mass * self.G * self.L_front / self.L
        F_nr = self.mass * self.G * self.L_rear / self.L

        # max forces
        max_F_f = mu * F_nf
        max_F_r = mu * F_nr

        # Compute the lateral "slip-slope"
        self.max_alpha_stat = 0.12
        slip_ratio_f = max_F_f / self.max_alpha_stat
        slip_ratio_r = max_F_r / self.max_alpha_stat

        if (u[1] == 0):
            slip_f = 0
            slip_r = 0
            F_yf = 0
            F_yr = 0
        else:
            slip_f = u[0] - np.arctan((x[0] + self.L_front*x[1]) / u[1])
            slip_r = np.arctan((self.L_rear*x[1] - x[0]) / u[1])

        if (slip_f < -self.max_alpha_stat):
            F_yf = -max_F_f
        elif (slip_f > self.max_alpha_stat):
            F_yf = max_F_f
        else:
            F_yf = slip_f * slip_ratio_f
        
        if (slip_r < -self.max_alpha_stat):
            F_yr = -max_F_r
        elif (slip_r > self.max_alpha_stat):
            F_yr = max_F_r
        else:
            F_yr = slip_r * slip_ratio_r

        return F_yf, F_yr
