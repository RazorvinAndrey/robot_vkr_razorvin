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
        self.path = [(x, y)]
        self.target_position = target_position
        self.step_size = 0.1
        self.d_min = np.inf
        self.following_boundary = False
        self.boundary_start = None
        self.stop = False
        self.angle_increment = np.deg2rad(10)  # Угол, на который робот поворачивается, если встречает препятствие

    def move(self, obstacles, robots):
        if not self.stop:
            if not self.following_boundary:
                # Двигаемся напрямую к цели
                self.move_towards_target(obstacles, robots)
            else:
                # Если уже следуем за границей, продолжаем делать это
                self.follow_boundary(obstacles, robots)

    def move_towards_target(self, obstacles, robots):
        new_x, new_y, target_reached = self.step_towards_target()
        if target_reached:
            self.stop = True
            return
        if not self.check_collision((new_x, new_y), obstacles, robots):
            self.x, self.y = new_x, new_y
            self.path.append((self.x, self.y))
        else:
            self.following_boundary = True
            self.boundary_start = (self.x, self.y)
            self.d_min = np.linalg.norm(np.array(self.target_position) - np.array((self.x, self.y)))
            # При столкновении, сначала поворачиваемся на месте
            self.angle += self.angle_increment

    def step_towards_target(self):
        direction = np.array(self.target_position) - np.array((self.x, self.y))
        distance = np.linalg.norm(direction)
        target_reached = distance < self.step_size
        step = direction / distance * min(self.step_size, distance)
        new_x = self.x + step[0]
        new_y = self.y + step[1]
        return new_x, new_y, target_reached

    def follow_boundary(self, obstacles, robots):
        # Изменяемый шаг угла для более плавного обхода препятствий
        angle_step = np.deg2rad(10)

        # Пытаемся двигаться в сторону препятствия
        self.angle += angle_step
        while True:
            new_x = self.x + np.cos(self.angle) * self.step_size
            new_y = self.y + np.sin(self.angle) * self.step_size

            # Проверяем свободен ли путь для движения
            if not self.check_collision((new_x, new_y), obstacles, robots):
                # Проверяем, не стали ли мы ближе к цели
                new_d_min = np.linalg.norm(np.array(self.target_position) - np.array((new_x, new_y)))
                if new_d_min < self.d_min:
                    self.d_min = new_d_min
                    # Если улучшили путь к цели, продолжаем движение к ней
                    if np.linalg.norm(np.array(self.boundary_start) - np.array((new_x, new_y))) < self.step_size:
                        self.following_boundary = False
                    break
                self.x, self.y = new_x, new_y
                self.path.append((self.x, self.y))
                return  # Нашли возможность движения, выходим из функции
            else:
                # Поворачиваемся немного, чтобы попробовать другое направление
                self.angle += angle_step

    def check_collision(self, position, obstacles, robots):
        for obstacle in obstacles + [(r.x, r.y) for r in robots if r != self]:
            if np.linalg.norm(np.array(position) - np.array(obstacle)) <= self.width / 2:
                return True
        return False

    def measure_distance(self, obstacles, robots):
        # Измеряем дистанцию до ближайшего препятствия в направлении лидара
        min_distance = self.lidar_range
        for obstacle in obstacles + [(r.x, r.y) for r in robots if r != self]:
            dx, dy = obstacle[0] - self.x, obstacle[1] - self.y
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
        return min_distance


class AnimatedSimulation:
    def __init__(self, num_robots, obst, init_area_center, init_area_radius, target_area_center, target_area_radius):
        self.robots = [
            Robot(init_area_center[0] + np.random.uniform(-init_area_radius, init_area_radius),
                  init_area_center[1] + np.random.uniform(-init_area_radius, init_area_radius),
                  np.random.rand() * 360,
                  (target_area_center[0] + np.random.uniform(-target_area_radius, target_area_radius),
                   target_area_center[1] + np.random.uniform(-target_area_radius, target_area_radius)))
            for _ in range(num_robots)]
        self.obstacles = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(obst)]
        self.fig, self.ax = plt.subplots()
        self.start_time = time.time()

    def animate(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        # Отображение препятствий
        for obstacle in self.obstacles:
            self.ax.plot(*obstacle, 'gx')

        # Отображение роботов, их пути, направления движения и целевых точек
        for robot in self.robots:
            if not robot.stop:
                robot.move(self.obstacles, self.robots)
            path_x, path_y = zip(*robot.path)
            self.ax.plot(*robot.target_position, 'rs')
            self.ax.plot(path_x, path_y, 'k-', linewidth=0.5)
            self.ax.plot(robot.x, robot.y, 'bo')

            # Расстояние, измеренное лидаром
            lidar_distance = robot.measure_distance(self.obstacles, self.robots)
            lidar_end_x = robot.x + np.cos(robot.angle) * lidar_distance
            lidar_end_y = robot.y + np.sin(robot.angle) * lidar_distance
            self.ax.plot([robot.x, lidar_end_x], [robot.y, lidar_end_y], 'r-')

    def run(self):
        anim = FuncAnimation(self.fig, self.animate, frames=np.arange(0, 200), interval=100)
        plt.show()


# Задаем параметры для генерации начальных и целевых позиций роботов
init_area_center = (2, 2)  # Центр области для начального расположения роботов
init_area_radius = 2  # Радиус начальной области
target_area_center = (7, 7)  # Центр области для целевого расположения роботов
target_area_radius = 2  # Радиус целевой области

# Запускаем симуляцию с заданным количеством роботов, препятствий и параметрами начального и целевого расположения
simulation = AnimatedSimulation(num_robots=10, obst=5, init_area_center=init_area_center, init_area_radius=init_area_radius, target_area_center=target_area_center, target_area_radius=target_area_radius)
simulation.run()
