import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
import random


pos = []
sec = 0


class Rectangle(patches.Rectangle):
    # Класс для описания препятствий
    def intersects(self, rect):
        return not (self.get_x() + self.get_width() < rect.get_x() or
                    self.get_x() > rect.get_x() + rect.get_width() or
                    self.get_y() + self.get_height() < rect.get_y() or
                    self.get_y() > rect.get_y() + rect.get_height())


class PheromoneMap:
    def __init__(self, width, height, decay_rate=0.1):
        self.map = np.zeros((height, width))
        self.decay_rate = decay_rate

    def update(self):
        self.map *= (1 - self.decay_rate)

    def add_pheromone(self, x, y, amount=1.0):
        if 0 <= int(x) < self.map.shape[1] and 0 <= int(y) < self.map.shape[0]:
            self.map[int(y), int(x)] += amount

    def get_pheromone_level(self, x, y):
        if 0 <= int(x) < self.map.shape[1] and 0 <= int(y) < self.map.shape[0]:
            return self.map[int(y), int(x)]
        return 0


class Robot:
    def __init__(self, x, y, target, obstacles, pheromone_map, robots=None):
        self.x = x
        self.y = y
        self.x0 = x
        self.y0 = y
        self.ang0 = 0
        self.target_position = target
        self.obstacles = obstacles
        self.pheromone_map = pheromone_map
        self.robots = robots if robots is not None else []
        self.path = [(x, y)]
        self.stop = False
        self.max_speed_per_step = 0.3
        self.width = 0.4
        self.angle = 0
        self.prev_x = x  # Предыдущее положение по x
        self.prev_y = y  # Предыдущее положение по y

    def move(self):
        if self.stop:
            return

        if np.hypot(self.x - self.target_position[0], self.y - self.target_position[1]) < self.max_speed_per_step / 2:
            self.x, self.y = self.target_position  # Перемещаем робота точно в целевую позицию
            self.x += random.uniform(-0.1, 0.1)
            self.y += random.uniform(-0.1, 0.1)
            self.stop = True
            return

        self.prev_x, self.prev_y = self.x, self.y  # Обновляем предыдущее положение
        self.choose_next_step()

    def choose_next_step(self):
        best_score = float('inf')
        best_move = (0, 0)

        # Увеличение детализации направлений движения
        step_angles = np.linspace(0, 360, 36, endpoint=False)  # 36 направлений
        for angle in step_angles:
            rad = np.radians(angle)
            dx = self.max_speed_per_step * np.cos(rad)
            dy = self.max_speed_per_step * np.sin(rad)
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < 10 and 0 <= ny < 10 and not self.check_collision(nx, ny):
                distance_to_target = np.hypot(self.target_position[0] - nx, self.target_position[1] - ny)
                if distance_to_target < best_score:
                    best_score = distance_to_target
                    best_move = (dx, dy)

        if best_move != (0, 0):
            self.x += best_move[0]
            self.y += best_move[1]
            self.path.append((self.x, self.y))
            self.pheromone_map.add_pheromone(self.x, self.y, 1.0)

        # Рассчитываем угол направления робота на основе изменения положения
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        self.angle = np.degrees(np.arctan2(dy, dx)) + random.uniform(-0.5, 0.5)

    def try_random_step(self):
        attempts = 0
        while attempts < 10:
            angle = np.radians(np.random.randint(0, 360))
            dx = self.max_speed_per_step * np.cos(angle)
            dy = self.max_speed_per_step * np.sin(angle)
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < 10 and 0 <= ny < 10 and not self.check_collision(nx, ny):
                self.x = nx
                self.y = ny
                self.path.append((self.x, self.y))
                self.pheromone_map.add_pheromone(self.x, self.y, 1.0)
                break
            attempts += 1

    def check_collision(self, x, y):
        robot_rect = patches.Rectangle((x - self.width / 2, y - self.width / 2), self.width, self.width)
        if any(ob.intersects(robot_rect) for ob in self.obstacles):
            return True
        return any(np.hypot(x - other.x, y - other.y) <= self.width for other in self.robots if other != self)

    def measure_distance(self):
        lidar_range = 5  # максимальный диапазон дальномера в метрах
        lidar_angle = np.radians(self.angle)
        min_distance = lidar_range  # начальное минимальное расстояние устанавливаем максимальным

        # Проверяем расстояние до препятствий в направлении дальномера
        for step in np.linspace(0, lidar_range, num=int(lidar_range/0.1)):
            check_x = self.x + np.cos(lidar_angle) * step
            check_y = self.y + np.sin(lidar_angle) * step
            for obstacle in self.obstacles:
                # Проверяем, пересекается ли точка с препятствием
                if obstacle.intersects(patches.Rectangle((check_x - self.width / 2, check_y - self.width / 2), self.width, self.width)):
                    min_distance = min(min_distance, step)  # Обновляем минимальное расстояние, если нашли ближайшее препятствие
                    break
            else:
                continue
            break

        lidar_end_x = self.x + np.cos(lidar_angle) * min_distance
        lidar_end_y = self.y + np.sin(lidar_angle) * min_distance

        return lidar_end_x, lidar_end_y


class AnimatedSimulation:
    def __init__(self, init_positions, target_positions, obstacle_params):
        self.start_time = time.time()  # Начальное время для замера времени выполнения
        self.all_robots_stopped = False
        self.width, self.height = 10, 10
        self.pheromone_map = PheromoneMap(self.width, self.height)
        self.obstacles = [Rectangle((obstacle['x'], obstacle['y']), obstacle['width'], obstacle['height']) for obstacle in obstacle_params]
        self.robots = [Robot(x, y, target, self.obstacles, self.pheromone_map) for (x, y), target in zip(init_positions, target_positions)]
        for robot in self.robots:
            robot.robots = self.robots
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')  # Устанавливаем одинаковый масштаб осей
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

    def animate(self):
        def update(frame):
            global pos, sec
            self.ax.clear()
            self.ax.set_aspect('equal')  # Сохраняем аспект при обновлении
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            for obstacle in self.obstacles:
                self.ax.add_patch(patches.Rectangle((obstacle.get_x(), obstacle.get_y()), obstacle.get_width(), obstacle.get_height()))
            active_robots = 0
            i = 0
            Vx = []
            Vy = []
            W = []
            for robot in self.robots:
                robot.move()
                path_x, path_y = zip(*robot.path)
                self.ax.plot(*robot.target_position, 'bs')
                self.ax.plot(path_x, path_y, 'k-', linewidth=1)
                self.ax.plot(robot.x, robot.y, 'ro')
                Vx.append((robot.x - robot.x0)/1)
                robot.x0 = robot.x
                Vy.append((robot.y - robot.y0)/1)
                robot.y0 = robot.y
                W.append((robot.angle - robot.ang0)/1)
                robot.ang0 = robot.angle

                # lidar_end_x, lidar_end_y = robot.measure_distance()
                # self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'g-')  # Рисуем луч дальномера

                if not robot.stop:
                    active_robots += 1
                i += 1
            pos.append([sum(Vx)/i, sum(Vy)/i, sum(W)/(100*i)])
            self.pheromone_map.update()

            if active_robots == 0 and not self.all_robots_stopped:
                self.all_robots_stopped = True
                end_time = time.time()
                total_time = end_time - self.start_time
                sec = total_time
                print(f"Все роботы остановились. Время выполнения: {total_time:.2f} секунд.")
        anim = FuncAnimation(self.fig, update, frames=np.arange(300), repeat=False)
        plt.show()


# Пример настройки
init_positions = [(0, 1), (2, 0), (1, 1), (0, 0)]
target_positions = [(8, 7), (7, 7), (8, 8), (8.5, 8)]
obstacles = [{'x': 4, 'y': 4, 'width': 1, 'height': 2}, {'x': 5, 'y': 1, 'width': 2, 'height': 1}, {'x': 6, 'y': 5, 'width': 1.5, 'height': 1.5}]
simulation = AnimatedSimulation(init_positions, target_positions, obstacles)
simulation.animate()
time_ = []
Vx = []
Vy = []
W = []
old_min, old_max = 0, len(pos)
new_min, new_max = 0, sec

old_range = old_max - old_min
new_range = new_max - new_min


for i in range(len(pos)):
    converted = ((i - old_min) * new_range / old_range) + new_min
    time_.append(converted)
    Vx.append(pos[i][0])
    Vy.append(pos[i][1])
    W.append(pos[i][2])

plt.suptitle("ACO")
plt.subplot(131)
plt.title("Vx")
plt.xlabel("time, s")
plt.ylabel("Vx, m/s")
plt.plot(time_, Vx)

plt.subplot(132)
plt.title("Vy")
plt.xlabel("time, s")
plt.ylabel("Vy, m/s")
plt.plot(time_, Vy)

plt.subplot(133)
plt.title("W")
plt.xlabel("time, s")
plt.ylabel("W, rad/s")
plt.plot(time_, W)

plt.show()
