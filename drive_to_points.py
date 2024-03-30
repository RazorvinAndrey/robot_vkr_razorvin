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
        self.width = 0.4
        self.length = 0.3
        self.path = [(x, y)]
        self.target_position = target_position
        self.steps = self.calculate_path_steps(target_position)
        self.step_index = 0
        self.stop = False
        self.stuck_steps = 0  # Количество шагов без изменения позиции
        self.last_position = (x, y)  # Последняя позиция для отслеживания изменений
        self.prev_x = x
        self.prev_y = y
        self.moved_since_last_check = False  # Изменил ли робот позицию с последней проверки

    def move(self, obstacles, robots):
        if self.step_index < len(self.steps) and not self.stop:
            vvv, www = self.steps[self.step_index]
            self.angle = (self.angle + www) % 360
            rad_angle = np.deg2rad(self.angle)

            proposed_x = self.x + vvv * np.cos(rad_angle)
            proposed_y = self.y + vvv * np.sin(rad_angle)

            # Проверка на столкновения
            if self.check_collision(proposed_x, proposed_y, obstacles, robots):
                self.avoid_collision(obstacles, robots)
            else:
                if (proposed_x, proposed_y) == self.last_position:
                    self.stuck_steps += 1
                else:
                    self.stuck_steps = 0
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))
                self.step_index += 1
                self.last_position = (self.x, self.y)
            if (self.x, self.y) != (self.prev_x, self.prev_y):  # Проверка на изменение позиции
                self.moved_since_last_check = True
            # Остановка, если робот не менял позицию достаточное количество шагов
            if self.stuck_steps > 10:  # Это значение можно настроить
                self.stop = True

    def check_collision(self, x, y, obstacles, robots):
        for obstacle in obstacles + [(r.x, r.y) for r in robots if r != self]:
            if np.hypot(x - obstacle[0], y - obstacle[1]) <= self.width / 2 + 0.1:
                return True
        return False

    def make_significant_move(self, obstacles, robots):
        # Attempt a larger movement in a random direction
        rad_angle = np.deg2rad(np.random.randint(0, 360))
        proposed_x = self.x + 0.5 * np.cos(rad_angle)  # Larger step size
        proposed_y = self.y + 0.5 * np.sin(rad_angle)
        if not self.check_collision(proposed_x, proposed_y, obstacles, robots):
            self.x, self.y = proposed_x, proposed_y
            self.path.append((self.x, self.y))
            self.steps = self.calculate_path_steps(self.target_position)
            self.step_index = 0

    def avoid_collision(self, obstacles, robots):
        # Attempt to find a clear direction to move
        for angle_offset in [30, -30, 60, -60, 90, -90, 120, -120, 180]:
            rad_angle = np.deg2rad(self.angle + angle_offset)
            proposed_x = self.x + 0.1 * np.cos(rad_angle)
            proposed_y = self.y + 0.1 * np.sin(rad_angle)
            if not self.check_collision(proposed_x, proposed_y, obstacles, robots):
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))
                self.steps = self.calculate_path_steps(self.target_position)  # Recalculate path to target
                self.step_index = 0
                return
        self.stuck_steps += 1  # Increment stuck steps if no clear direction is found

        # If the robot is stuck for too many steps, make a more significant move
        if self.stuck_steps >= 10:  # This threshold can be adjusted
            self.make_significant_move(obstacles, robots)
            self.stuck_steps = 0  # Reset the counter after making a significant move

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
            if abs(angle_diff) > 1:
                steps.append((0, angle_diff))
            else:
                steps.append((min(0.2, distance), 0))

            current_angle += np.deg2rad(angle_diff)
            current_x += np.cos(current_angle) * min(0.2, distance)
            current_y += np.sin(current_angle) * min(0.2, distance)

        return steps

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


class AnimatedSimulation:
    def __init__(self, num_robots, obst, target_position, init_area_center, init_area_radius):
        self.robots = [
            Robot(init_area_center[0] + np.random.uniform(-init_area_radius, init_area_radius),
                  init_area_center[1] + np.random.uniform(-init_area_radius, init_area_radius),
                  np.random.rand() * 360, target_position)
            for _ in range(num_robots)]
        self.obstacles = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(obst)]
        self.fig, self.ax = plt.subplots()
        self.start_time = time.time()
        self.all_robots_stopped = False

    def step(self):
        all_stopped = True
        for robot in self.robots:
            if not robot.stop:
                robot.move(self.obstacles, self.robots)
                if robot.step_index < len(robot.steps):
                    all_stopped = False

        if all_stopped and not self.all_robots_stopped:
            self.all_robots_stopped = True
            end_time = time.time()
            print(f"Все роботы остановились. Время выполнения: {end_time - self.start_time:.2f} секунд.")

    def animate(self):
        def update(frame):
            nonlocal active_robots
            nonlocal positions_last_frame

            active_robots = 0
            positions_this_frame = []

            for robot in self.robots:
                if not robot.stop:
                    robot.move(self.obstacles, self.robots)
                    if robot.step_index < len(robot.steps):
                        active_robots += 1
                        positions_this_frame.append((robot.x, robot.y))

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
            if positions_this_frame == positions_last_frame:
                self.steps_without_movement += 1
            else:
                self.steps_without_movement = 0

            positions_last_frame = positions_this_frame

            if self.steps_without_movement >= 10 or active_robots == 0:  # Условие остановки анимации
                end_time = time.time()
                print(f"Все роботы остановились или застряли. Время выполнения: {end_time - self.start_time:.2f} секунд.")
                plt.close(self.fig)
            # Закрываем анимацию, если все роботы остановились
            # if active_robots == 0:
            #     plt.close()

        active_robots = len(self.robots)
        positions_last_frame = []
        anim = FuncAnimation(self.fig, update, frames=np.arange(100), repeat=False)
        plt.show()


# Задаем параметры для генерации начальных позиций роботов
init_area_center = (2, 2)  # Центр области для начального расположения роботов
init_area_radius = 2  # Радиус области

# Запускаем симуляцию с заданным количеством роботов, препятствий и параметрами начального расположения
target_position = (5, 5)
simulation = AnimatedSimulation(num_robots=5, obst=20, target_position=target_position, init_area_center=init_area_center, init_area_radius=init_area_radius)
simulation.animate()
