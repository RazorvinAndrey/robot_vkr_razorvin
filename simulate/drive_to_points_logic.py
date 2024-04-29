import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time


class Rectangle(patches.Rectangle):
    def distance(self, point):
        closest_x = max(self.get_x(), min(point[0], self.get_x() + self.get_width()))
        closest_y = max(self.get_y(), min(point[1], self.get_y() + self.get_height()))
        return np.sqrt((point[0] - closest_x) ** 2 + (point[1] - closest_y) ** 2)

    def intersects(self, other_rect):
        return self.get_x() < other_rect.get_x() + other_rect.get_width() and \
               self.get_x() + self.get_width() > other_rect.get_x() and \
               self.get_y() < other_rect.get_y() + other_rect.get_height() and \
               self.get_y() + self.get_height() > other_rect.get_y()


class Robot:
    def __init__(self, x, y, angle, target, obstacles, robots=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.lidar_range = 5
        self.width = 0.4
        self.max_speed_per_step = 0.5  # максимальная скорость 0.5 метра в секунду
        self.path = [(x, y)]
        self.target_position = target
        self.obstacles = obstacles
        self.robots = robots if robots is not None else []  # Добавляем проверку на robots
        self.steps = self.calculate_path_steps(self.target_position)
        self.step_index = 0
        self.stop = False
        self.stuck_steps = 0
        self.last_position = (x, y)

    def move(self):
        if self.step_index < len(self.steps) and not self.stop:
            distance, angle_change = self.steps[self.step_index]
            self.angle = (self.angle + angle_change) % 360
            rad_angle = np.deg2rad(self.angle)

            proposed_x = self.x + distance * np.cos(rad_angle)
            proposed_y = self.y + distance * np.sin(rad_angle)

            if not self.check_collision(proposed_x, proposed_y, self.robots):
                if (proposed_x, proposed_y) == self.last_position:
                    self.stuck_steps += 1
                else:
                    self.stuck_steps = 0
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))
                self.step_index += 1
                self.last_position = (self.x, self.y)
            else:
                self.avoid_collision()

            if self.stuck_steps > 10:
                self.stop = True

    def check_collision(self, x, y, robots):
        for obstacle in self.obstacles:
            if obstacle.intersects(Rectangle((x - self.width/2, y - self.width/2), self.width, self.width)):
                return True
        for obstacle in [(r.x, r.y) for r in robots if r != self]:
            if np.hypot(x - obstacle[0], y - obstacle[1]) <= self.width / 2 + 0.1:
                return True
        return False

    def avoid_collision(self):
        for angle_offset in [30, -30, 60, -60, 90, -90, 120, -120, 180]:
            rad_angle = np.deg2rad(self.angle + angle_offset)
            proposed_x = self.x + 0.3 * np.cos(rad_angle)
            proposed_y = self.y + 0.3 * np.sin(rad_angle)
            if not self.check_collision(proposed_x, proposed_y, self.robots):
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))
                self.steps = self.calculate_path_steps(self.target_position)
                self.step_index = 0
                return
        self.stuck_steps += 1

    def measure_distance(self):
        lidar_range = self.lidar_range
        lidar_angle = self.angle
        min_distance = lidar_range  # Устанавливаем максимальное расстояние дальномера как минимальное найденное расстояние до препятствия

        for angle_step in range(-5, 6):  # Проверяем углы от -5 до 5 градусов относительно текущего направления робота для учета небольших расхождений
            check_angle = np.deg2rad(self.angle + angle_step)
            for step in np.linspace(0, lidar_range, num=int(lidar_range/0.1)):
                check_x = self.x + np.cos(check_angle) * step
                check_y = self.y + np.sin(check_angle) * step
                for obstacle in self.obstacles:
                    if obstacle.distance((check_x, check_y)) < self.width / 2:
                        min_distance = min(min_distance, step)  # Обновляем минимальное расстояние, если найдено более близкое препятствие
                        break
                else:
                    continue
                break

        lidar_end_x = self.x + np.cos(np.deg2rad(lidar_angle)) * min_distance
        lidar_end_y = self.y + np.sin(np.deg2rad(lidar_angle)) * min_distance

        return lidar_end_x, lidar_end_y

    def calculate_path_steps(self, target_position):
        steps = []
        current_x, current_y = self.x, self.y
        current_angle = np.deg2rad(self.angle)
        target_x, target_y = target_position

        while True:
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.01:
                break

            target_angle = np.arctan2(dy, dx)
            angle_diff = np.rad2deg(np.mod(target_angle - current_angle + np.pi, 2 * np.pi) - np.pi)
            if abs(angle_diff) > 1:
                steps.append((0, angle_diff))
                distance = 0
            else:
                if distance <= self.max_speed_per_step:
                    steps.append((distance, 0))
                    break
                else:
                    steps.append((self.max_speed_per_step, 0))

            current_angle += np.deg2rad(angle_diff)
            current_x += np.cos(current_angle) * min(self.max_speed_per_step, distance)
            current_y += np.sin(current_angle) * min(self.max_speed_per_step, distance)

        return steps

    def avoid_obstacles_and_robots(self):
        for other_robot in self.robots:
            if other_robot != self:
                # Проверяем расстояние между текущим и другим роботом
                dx = other_robot.x - self.x
                dy = other_robot.y - self.y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < 0.3:  # Учитываем положение другого робота только если он на расстоянии меньше 0.3 метра
                    self.avoid_robot_collision(other_robot)
                    self.avoid_collision()
                    # Обновляем угол движения после избегания столкновений с другими роботами
                    self.angle = np.degrees(np.arctan2(self.steps[self.step_index][1], 1))

    def avoid_robot_collision(self, other_robot):
        min_distance = 0.3  # Минимальное расстояние между роботами
        dx = other_robot.x - self.x
        dy = other_robot.y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance < min_distance:
            # Вычисляем новый угол, чтобы роботы не сталкивались
            target_angle = np.arctan2(dy, dx)
            angle_diff = np.rad2deg(np.mod(target_angle - np.deg2rad(self.angle) + np.pi, 2 * np.pi) - np.pi)
            self.angle = (self.angle + angle_diff) % 360

            # Корректируем позицию робота, чтобы он не наезжал на другого робота
            proposed_x = self.x - min_distance * np.cos(target_angle)
            proposed_y = self.y - min_distance * np.sin(target_angle)
            if not self.check_collision(proposed_x, proposed_y, self.robots):
                self.x, self.y = proposed_x, proposed_y


class AnimatedSimulation:
    def __init__(self, init_positions, target_positions, obstacle_params):
        self.robots = []  # Создаем пустой список для роботов
        for (x, y), target in zip(init_positions, target_positions):
            robot = Robot(x, y, np.random.rand() * 360, target, self.create_obstacles(obstacle_params))
            robot.robots = self.robots  # Передаем список роботов каждому роботу
            self.robots.append(robot)  # Добавляем созданный робот в список self.robots
        self.fig, self.ax = plt.subplots()
        self.width, self.height = 10, 10
        self.ax.set_aspect('equal')  # Устанавливаем одинаковый масштаб осей
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.start_time = time.time()
        self.all_robots_stopped = False
        self.steps_without_movement = 0

    def create_obstacles(self, obstacle_params):
        obstacles = []
        for obstacle in obstacle_params:
            obstacles.append(Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height']))
        return obstacles

    def animate(self):
        def update(frame):
            nonlocal active_robots

            active_robots = 0

            for robot in self.robots:
                if not robot.stop:
                    robot.move()
                    if robot.step_index < len(robot.steps):
                        active_robots += 1

            self.ax.clear()
            self.ax.set_aspect('equal')  # Сохраняем аспект при обновлении
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)

            for obstacle in robot.obstacles:
                self.ax.add_patch(obstacle)

            for robot in self.robots:
                path_x, path_y = zip(*robot.path)
                self.ax.plot(*robot.target_position, 'rs')
                self.ax.plot(path_x, path_y, 'k-', linewidth=0.5)
                self.ax.plot(robot.x, robot.y, 'bo')
                lidar_end_x, lidar_end_y = robot.measure_distance()
                self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'g-')

                # Проверка столкновений с другими роботами
                robot.avoid_obstacles_and_robots()

            if active_robots == 0 and not self.all_robots_stopped:
                end_time = time.time()
                print(f"All robots have stopped or reached their targets. Execution time: {end_time - self.start_time:.2f} seconds.")
                self.all_robots_stopped = True

        active_robots = len(self.robots)
        anim = FuncAnimation(self.fig, update, frames=np.arange(300), repeat=False)
        plt.show()


# Пример настройки
init_positions = [(0, 1), (2, 0), (1, 1), (0, 0)]
target_positions = [(8, 7), (7, 7), (8, 8), (8.5, 8)]
obstacles = [{'x': 4, 'y': 4, 'width': 1, 'height': 2}, {'x': 5, 'y': 1, 'width': 2, 'height': 1}, {'x': 6, 'y': 5, 'width': 1.5, 'height': 1.5}]
simulation = AnimatedSimulation(init_positions, target_positions, obstacles)
simulation.animate()
