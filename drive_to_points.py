import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class Robot:
    def __init__(self, x, y, angle, target_position):
        self.x = x
        self.y = y
        self.angle = angle
        self.lidar_range = 5
        self.width = 0.2
        self.length = 0.3
        self.path = [(x, y)]  # Starting position
        self.steps = self.calculate_path_steps(target_position)
        self.step_index = 0  # Current step index
        self.stop = False

    def move(self, obstacles, robots):
        if self.step_index < len(self.steps):
            vvv, www = self.steps[self.step_index]
            self.angle = (self.angle + www) % 360
            rad_angle = np.deg2rad(self.angle)

            proposed_x = self.x + vvv * np.cos(rad_angle)
            proposed_y = self.y + vvv * np.sin(rad_angle)

            if not self.check_collision(proposed_x, proposed_y, obstacles, robots):
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))

            self.step_index += 1

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
        # This avoids modifying the original obstacles list
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

    def calculate_path_steps(self, target_position):
        steps = []
        current_x, current_y = self.x, self.y
        current_angle = np.deg2rad(self.angle)
        target_x, target_y = target_position

        while True:
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.05:  # Close enough to target
                break

            target_angle = np.arctan2(dy, dx)
            angle_diff = np.rad2deg(np.mod(target_angle - current_angle + np.pi, 2 * np.pi) - np.pi)
            if abs(angle_diff) > 1:  # Need to rotate
                steps.append((0, angle_diff))
            else:  # Move forward
                steps.append((min(0.2, distance), 0))

            # Dummy update to current position and angle for the next iteration
            current_angle += np.deg2rad(angle_diff)
            current_x += np.cos(current_angle) * min(0.2, distance)
            current_y += np.sin(current_angle) * min(0.2, distance)

        return steps


class AnimatedSimulation:
    def __init__(self, num_robots, obst, target_position):
        self.robots = [Robot(np.random.rand() * 10, np.random.rand() * 10, np.random.rand() * 360, target_position) for _ in range(num_robots)]
        self.obstacles = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(obst)]
        self.fig, self.ax = plt.subplots()
        self.start_time = time.time()
        self.all_robots_stopped = False

    def step(self):
        active_robots = 0
        for robot in self.robots:
            if robot.step_index < len(robot.steps):  # Проверяем, не завершил ли робот свой маршрут
                active_robots += 1
                robot.move(self.obstacles, self.robots)
            else:
                robot.stop = True
        if active_robots == 0 and not self.all_robots_stopped:
            self.all_robots_stopped = True
            end_time = time.time()
            print(f"Все роботы остановились. Время выполнения: {end_time - self.start_time:.2f} секунд.")
        return active_robots

    def animate(self):
        def update(frame):
            active_robots = self.step()
            self.ax.clear()
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)

            for obstacle in self.obstacles:
                self.ax.plot(*obstacle, 'gx')

            for robot in self.robots:
                path_x, path_y = zip(*robot.path)
                self.ax.plot(path_x, path_y, 'k-', linewidth=0.5)
                self.ax.plot(robot.x, robot.y, 'bo')

                # Расчет и отображение лидарных данных
                lidar_distance = robot.measure_distance(self.obstacles, self.robots)
                lidar_end_x = robot.x + np.cos(np.deg2rad(robot.angle)) * lidar_distance
                lidar_end_y = robot.y + np.sin(np.deg2rad(robot.angle)) * lidar_distance
                self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'r-')

            # Закрываем анимацию, если все роботы остановились
            # if active_robots == 0:
            #     plt.close()

        anim = FuncAnimation(self.fig, update, frames=np.arange(100), repeat=False)
        plt.show()

# Запускаем симуляцию с заданным количеством роботов и препятствий, двигаясь к целевой точке.
target_position = (5, 5)
simulation = AnimatedSimulation(num_robots=5, obst=20, target_position=target_position)
simulation.animate()

