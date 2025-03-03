

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator # type: ignore
from scipy.signal import TransferFunction # type: ignore
import json
import os 

class SplinePath:
    def __init__(self, waypoints, v_long_profile=None, closed=True, t_fine=None, name=None):
        self.waypoints = waypoints
        self.closed = closed
        self.name = name

        self.t_points = np.linspace(0, 1, len(waypoints))
        if closed:
            bc_type = "periodic"
        else:
            bc_type = "clamped"

        self.spline_x = CubicSpline(self.t_points, waypoints[:, 0], bc_type=bc_type)
        self.spline_y = CubicSpline(self.t_points, waypoints[:, 1], bc_type=bc_type)

        if t_fine is None:
            t_fine = np.linspace(0, 1, 10000)
        else:
            t_fine = t_fine / t_fine.max()
        self.t_fine = t_fine
        self.x_fine = self.spline_x(t_fine)
        self.y_fine = self.spline_y(t_fine)

        self.dxdt_fine = self.spline_x(t_fine, 1)
        self.dydt_fine = self.spline_y(t_fine, 1)
        self.psi_fine = np.arctan2(self.dydt_fine, self.dxdt_fine)
        
        self.curvature = self.get_curvature(t_fine)
        self.length = self.calc_length()

        self.PT1_velocity = TransferFunction([3], [0.1, 1])
        if v_long_profile is None:
            vl = np.ones_like(t_fine)
        else:
            interp = PchipInterpolator(v_long_profile['theta'], v_long_profile['vl'])
            vl = interp(t_fine)
        self.v_long_profile = vl

    def get_waypoint(self, t):
        x = self.spline_x(t)
        y = self.spline_y(t)
        return x, y
    
    def get_tangent(self, t):
        dx = self.spline_x(t, 1) # call to CubicSpline also evaluates the derivative if the second argument is provided
        dy = self.spline_y(t, 1)
        return dx, dy
    
    def get_curvature(self, t):
        dx = self.spline_x(t, 1)
        ddx = self.spline_x(t, 2)
        dy = self.spline_y(t, 1)
        ddy = self.spline_y(t, 2)

        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        return curvature
    
    def plot(self, title=None):
        plt.figure()
        plt.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c="r")
        plt.plot(self.x_fine, self.y_fine, label="Spline")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.grid()
        plt.axis("equal")
        plt.show()

    def calc_length(self):
        """ Calculate the length of the path. """
        dx = np.diff(self.x_fine)
        dy = np.diff(self.y_fine)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        return np.sum(segment_lengths)
    
    def get_trajectory(self, T_end=0, dt=0.05): # T_end is the total time for the trajectory
        if T_end == 0:
            # self.T_end = 1.2*self.length / np.mean(self.v_long_profile)
            self.T_end = np.trapz(1 / self.v_long_profile, dx=(self.t_fine[1] - self.t_fine[0])) * self.length
        else:
            self.T_end = T_end
        TG = TrajectoryGenerator(self, self.T_end, dt)
        trajectory = TG.generate_trajectory()
        return trajectory


class TrajectoryGenerator:
    def __init__(self, spline_path, T_end, dt):
        self.spline_path = spline_path
        self.T_end = T_end
        self.dt = dt

    def generate_trajectory(self):
        path_length = self.spline_path.length
        v_profile = self.spline_path.v_long_profile  # velocity profile

        # Total time for traversing the path once
        t_total = np.trapz(1 / v_profile, dx=(self.spline_path.t_fine[1] - self.spline_path.t_fine[0])) * path_length

        # Initialize trajectory storage
        num_steps = int(self.T_end / self.dt)
        time_array = np.linspace(0, self.T_end, num_steps)
        trajectory = {"time": [], "x": [], "y": [], "psi": [], "vl": [], "kappa": []}

        # Keep track of cumulative distance traveled
        cumulative_distance = 0

        # Loop over the time steps
        for t in time_array:
            # Calculate the distance traveled up to this point
            current_velocity = np.interp(cumulative_distance / path_length, self.spline_path.t_fine, v_profile)
            distance_step = current_velocity * self.dt
            cumulative_distance += distance_step

            # Wrap cumulative distance if it exceeds path length (for multiple traversals)
            while cumulative_distance > path_length:
                cumulative_distance -= path_length

            # Get normalized position along the path
            normalized_t = cumulative_distance / path_length

            # Interpolate position and heading from the spline
            x, y = self.spline_path.get_waypoint(normalized_t)
            psi = np.interp(normalized_t, self.spline_path.t_fine, self.spline_path.psi_fine)

            # interpolate the velocity profile
            vl = np.interp(normalized_t, self.spline_path.t_fine, v_profile)
            # interpolate the curvature
            kappa = np.interp(normalized_t, self.spline_path.t_fine, self.spline_path.curvature)

            # Store the results in the trajectory
            trajectory["time"].append(t)
            trajectory["x"].append(x)
            trajectory["y"].append(y)
            trajectory["psi"].append(psi)
            trajectory["vl"].append(vl)
            trajectory["kappa"].append(kappa)

        return trajectory



def load_paths_from_json(file_path):
    with open(file_path, 'r') as f:
        paths = json.load(f)
    return paths

def create_spline_paths(file, real_world=False):
    paths = load_paths_from_json(file)
    spline_paths = []
    velo_factor = 1
    path_factor = 1
    if real_world:
        velo_factor = 30
        path_factor = 100
    for path in paths:
        waypoints = np.vstack((path['path']['x'], path['path']['y'])).T * path_factor
        waypoints = np.vstack((waypoints, waypoints[0])) # close the path
        vLong_profile = {'vl': np.array(path['path']['vLongPoints']) * velo_factor, 'theta': path['path']['vTheta']}
        spline_path = SplinePath(waypoints, v_long_profile=vLong_profile, closed=True, name=path['name'])
        spline_paths.append(spline_path)
    return spline_paths

def main(real_world=False):

    velo_factor = 1
    path_factor = 1
    if real_world:
        velo_factor = 30
        path_factor = 100

    if os.getcwd().split('\\')[-1] == 'paper_FedFF':
        path_file = 'paths.json'

    with open(path_file, 'r') as f:
        paths = json.load(f)

    for path in paths[5:6]:
        print(f"Path: {path['name']}")
        print(f"\tLength of waypoints x: {len(path['path']['x'])}")
        print(f"\tLength of waypoints y: {len(path['path']['y'])}")
        print(f"\tLenght of vLongPoints: {len(path['path']['vLongPoints'])}")
        print(f"\tLenght of vTheta: {len(path['path']['vTheta'])}")
        waypoints = np.vstack((path['path']['x'], path['path']['y'])).T * path_factor
        waypoints = np.vstack((waypoints, waypoints[0])) # close the path
        vLong_profile = {'vl': np.array(path['path']['vLongPoints']) * velo_factor, 'theta': path['path']['vTheta']}

        spline_path = SplinePath(waypoints, v_long_profile=vLong_profile, closed=True)
        spline_path.plot(title=path['name'] + f" lenght = {spline_path.length:.2f}m")
        print(f"Path length: {spline_path.length:.2f}")
        plt.pause(0.1)

    T_end = 200
    dt = 0.05
    trajectory_generator = TrajectoryGenerator(spline_path, T_end, dt)
    trajectory = trajectory_generator.generate_trajectory()

    # Plot the trajectory
    plt.figure()
    plt.plot(trajectory["time"], trajectory["y"], label="y")
    plt.plot(trajectory["time"], trajectory["x"], label="x")
    plt.plot(trajectory["time"], trajectory["psi"], label="psi")
    plt.plot(trajectory['time'], trajectory['vl'], label='vl')
    plt.plot(trajectory['time'], trajectory['kappa'], label='kappa')
    plt.title("Generated Trajectory")
    plt.xlabel("time ")
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__ == "__main__":
    main(real_world=True)