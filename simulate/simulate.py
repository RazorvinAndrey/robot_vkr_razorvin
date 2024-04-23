import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Robot:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.lidar_range = 5
        self.width = 0.2
        self.length = 0.3
        self.path = [(x, y)]  # Starting position

    def move(self, speed, turn_angle, obstacles, robots):
        proposed_speed = min(max(speed, -0.2), 0.2)
        self.angle = (self.angle + turn_angle) % 360
        rad_angle = np.deg2rad(self.angle)

        new_x = self.x + proposed_speed * np.cos(rad_angle)
        new_y = self.y + proposed_speed * np.sin(rad_angle)

        if not self.check_collision(new_x, new_y, obstacles, robots):
            self.x = new_x
            self.y = new_y
            self.path.append((self.x, self.y))

    def check_collision(self, x, y, obstacles, robots):
        for obstacle in obstacles:
            if np.hypot(x - obstacle[0], y - obstacle[1]) <= self.width / 2:
                return True
        for robot in robots:
            if robot != self and np.hypot(x - robot.x, y - robot.y) <= self.width:
                return True
        return False

    def measure_distance(self, obstacles, robots):
        min_distance = self.lidar_range
        # Create a temporary list combining obstacles with other robots' positions
        temp_obstacles = obstacles[:] + [(r.x, r.y) for r in robots if r != self]

        for obstacle in temp_obstacles:
            dx, dy = obstacle[0] - self.x, obstacle[1] - self.y
            distance = np.sqrt(dx**2 + dy**2)
            obstacle_angle = np.rad2deg(np.arctan2(dy, dx)) % 360

            angle_diff = abs(obstacle_angle - self.angle) % 360
            if angle_diff < 10 or angle_diff > 350:  # Narrow field of view for simplicity
                if distance < min_distance:
                    min_distance = distance

        return min_distance


class AnimatedSimulation:
    def __init__(self, num_robots, steps, obst):
        self.robots = [Robot(np.random.rand() * 10, np.random.rand() * 10, np.random.randint(360)) for _ in range(num_robots)]
        self.obstacles = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(obst)]
        self.fig, self.ax = plt.subplots()
        self.steps = steps

    def step(self):
        for robot in self.robots:
            robot.move(np.random.uniform(-0.2, 0.2), np.random.randint(-30, 30), self.obstacles, self.robots)

    def animate(self):
        def update(frame):
            self.step()
            self.ax.clear()
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)

            # Plot static obstacles
            for obstacle in self.obstacles:
                self.ax.plot(*obstacle, 'gx')

            # Plot robots and their lidar measurements
            for robot in self.robots:
                path_x, path_y = zip(*robot.path)
                self.ax.plot(path_x, path_y, 'k-', linewidth=0.5)  # Robot path
                self.ax.plot(robot.x, robot.y, 'bo')  # Robot position

                # Calculate lidar end point based on measurements
                lidar_distance = robot.measure_distance(self.obstacles, self.robots)
                lidar_end_x = robot.x + np.cos(np.deg2rad(robot.angle)) * lidar_distance
                lidar_end_y = robot.y + np.sin(np.deg2rad(robot.angle)) * lidar_distance
                self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'r-')  # Lidar line

        anim = FuncAnimation(self.fig, update, frames=range(self.steps), repeat=False)
        plt.show()


simulation = AnimatedSimulation(num_robots=20, steps=50, obst=20)
simulation.animate()
