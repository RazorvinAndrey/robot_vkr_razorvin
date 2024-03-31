import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class Robot:
    def __init__(self, x, y, angle, area_center, area_radius):
        self.x = x
        self.y = y
        self.angle = angle
        self.lidar_range = 5
        self.width = 0.4
        self.path = [(x, y)]
        # Выбираем случайную целевую позицию внутри заданной области
        self.target_position = (area_center[0] + np.random.uniform(-area_radius, area_radius),
                                area_center[1] + np.random.uniform(-area_radius, area_radius))
        self.steps = self.calculate_path_steps(self.target_position)
        self.step_index = 0
        self.stop = False
        self.stuck_steps = 0
        self.last_position = (x, y)
        self.prev_x = x
        self.prev_y = y

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
            # Остановка, если робот не менял позицию достаточное количество шагов
            if self.stuck_steps > 10:  # Это значение можно настроить
                self.stop = True

    def check_collision(self, x, y, obstacles, robots):
        # Проверка на столкновения с роботами и препятствиями
        for obstacle in obstacles + [(r.x, r.y) for r in robots if r != self]:
            if np.hypot(x - obstacle[0], y - obstacle[1]) <= self.width / 2 + 0.1:
                return True
        return False

    def make_significant_move(self, obstacles, robots):
        # Попытка движения в случайном направлении
        rad_angle = np.deg2rad(np.random.randint(0, 360))
        proposed_x = self.x + 0.5 * np.cos(rad_angle)  # Размер шага
        proposed_y = self.y + 0.5 * np.sin(rad_angle)
        if not self.check_collision(proposed_x, proposed_y, obstacles, robots):
            self.x, self.y = proposed_x, proposed_y
            self.path.append((self.x, self.y))
            self.steps = self.calculate_path_steps(self.target_position)
            self.step_index = 0

    def avoid_collision(self, obstacles, robots):
        # Попытка найти свободное направление без препятствий
        for angle_offset in [30, -30, 60, -60, 90, -90, 120, -120, 180]:
            rad_angle = np.deg2rad(self.angle + angle_offset)
            proposed_x = self.x + 0.1 * np.cos(rad_angle)
            proposed_y = self.y + 0.1 * np.sin(rad_angle)
            if not self.check_collision(proposed_x, proposed_y, obstacles, robots):
                self.x, self.y = proposed_x, proposed_y
                self.path.append((self.x, self.y))
                self.steps = self.calculate_path_steps(self.target_position)  # Пересчёт пути до цели
                self.step_index = 0
                return
        self.stuck_steps += 1  # Счётчик шагов без движения

        # Если робот застрял в течении нескольких шагов, начинаем движение в случайном направлении
        if self.stuck_steps >= 10:
            self.make_significant_move(obstacles, robots)
            self.stuck_steps = 0

    def calculate_path_steps(self, target_position):
        # Расчёт пути движения к цели
        steps = []
        current_x, current_y = self.x, self.y
        current_angle = np.deg2rad(self.angle)
        target_x, target_y = target_position

        while True:
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.05:  # Окрестность целевой точки
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
        # Добавим известные позиции роботов к списку препятствий, чтобы симулировать работу дальномера
        temp_obstacles = obstacles[:] + [(r.x, r.y) for r in robots if r != self]

        for obstacle in temp_obstacles:
            dx, dy = obstacle[0] - self.x, obstacle[1] - self.y
            distance = np.sqrt(dx**2 + dy**2)
            obstacle_angle = np.rad2deg(np.arctan2(dy, dx)) % 360

            angle_diff = abs(obstacle_angle - self.angle) % 360
            if angle_diff < 10 or angle_diff > 350:  # Поле зрения дальномера
                if distance < min_distance:
                    min_distance = distance

        return min_distance


class AnimatedSimulation:
    def __init__(self, num_robots, obst, init_area_center, init_area_radius, target_area_center, target_area_radius):
        self.robots = [
            Robot(init_area_center[0] + np.random.uniform(-init_area_radius, init_area_radius),
                  init_area_center[1] + np.random.uniform(-init_area_radius, init_area_radius),
                  np.random.rand() * 360, target_area_center, target_area_radius)
            for _ in range(num_robots)]
        self.obstacles = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(obst)]
        self.fig, self.ax = plt.subplots()
        self.start_time = time.time()
        self.all_robots_stopped = False

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

            # Отображение препятствий
            for obstacle in self.obstacles:
                self.ax.plot(*obstacle, 'gx')
            # Отображение роботов, их пути, направления движения и целевых точек
            for robot in self.robots:
                path_x, path_y = zip(*robot.path)
                self.ax.plot(*robot.target_position, 'rs')
                self.ax.plot(path_x, path_y, 'k-', linewidth=0.5)
                self.ax.plot(robot.x, robot.y, 'bo')
                lidar_distance = robot.measure_distance(self.obstacles, self.robots)
                lidar_end_x = robot.x + np.cos(np.deg2rad(robot.angle)) * lidar_distance
                lidar_end_y = robot.y + np.sin(np.deg2rad(robot.angle)) * lidar_distance
                self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'r-')

            if positions_this_frame == positions_last_frame:
                self.steps_without_movement += 1
            else:
                self.steps_without_movement = 0

            positions_last_frame = positions_this_frame

            # Остановка программы
            if (self.steps_without_movement >= 10 or active_robots == 0) and not self.all_robots_stopped:
                end_time = time.time()
                print(f"Все роботы остановились или достигли своих целей. Время выполнения: {end_time - self.start_time:.2f} секунд.")
                self.all_robots_stopped = True
                # plt.close(self.fig)

        active_robots = len(self.robots)
        positions_last_frame = []
        anim = FuncAnimation(self.fig, update, frames=np.arange(300), repeat=False)
        plt.show()


# Задаем параметры для генерации начальных и целевых позиций роботов
init_area_center = (2, 2)  # Центр области для начального расположения роботов
init_area_radius = 2  # Радиус начальной области
target_area_center = (7, 7)  # Центр области для целевого расположения роботов
target_area_radius = 2  # Радиус целевой области

# Запускаем симуляцию с заданным количеством роботов, препятствий и параметрами начального и целевого расположения
simulation = AnimatedSimulation(num_robots=10, obst=2, init_area_center=init_area_center, init_area_radius=init_area_radius, target_area_center=target_area_center, target_area_radius=target_area_radius)
simulation.animate()
