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
from scipy import signal # type: ignore

from controller_models import TrajectoryFollower, SteeringRateController
    
class BicycleModel:
    """ Describes a bicycle model in 2D. """
    def __init__(self, 
                 initial_state: dict, 
                 cfg: dict):
        """
        Describes a bicycle model in 2D.
        """
        np.random.seed(cfg['simulation']['random_seed'])

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