import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time


class PSORobot:
    def __init__(self, x, y, target, obstacles):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.rand(2) * 0.1  # Инициализируем небольшую случайную начальную скорость
        self.angle = np.arctan2(self.velocity[1], self.velocity[0])
        self.target = np.array(target, dtype=float)
        self.path = [tuple(self.position)]
        self.max_speed = 0.5
        self.max_angular_speed = np.pi / 2
        self.obstacles = obstacles
        self.lidar_range = 5
        self.all_robots = None
        self.at_target = False

    def update_all_robots(self, robots):
        self.all_robots = robots

    def update_velocity(self, global_best_position, delta_time=1):
        if not self.at_target:
            inertia = 0.2
            cognitive = 3.5
            social = 3
            r1, r2 = np.random.rand(), np.random.rand()

            cognitive_velocity = cognitive * r1 * (self.target - self.position)
            social_velocity = social * r2 * (global_best_position - self.position)

            velocity = inertia * self.velocity + cognitive_velocity + social_velocity
            speed = np.linalg.norm(velocity)

            if speed > self.max_speed:
                velocity = velocity * (self.max_speed / speed)
            self.velocity = velocity
            self.update_angle(delta_time)

    def update_angle(self, delta_time):
        if np.linalg.norm(self.velocity) > 0:  # Убеждаемся, что скорость не нулевая
            target_angle = np.arctan2(self.velocity[1], self.velocity[0])
            angle_diff = (target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
            max_angle_change = self.max_angular_speed * delta_time
            angle_diff = np.clip(angle_diff, -max_angle_change, max_angle_change)
            self.angle += angle_diff

    def move(self, delta_time=1):
        self.check_if_at_target()
        if not self.at_target:
            if self.avoid_obstacles_and_robots() or True:  # Убедитесь, что условие для движения правильное
                self.position += self.velocity * delta_time
                self.path.append(tuple(self.position))

    def check_if_at_target(self):
        target_radius = 0.2
        if np.linalg.norm(self.position - self.target) < target_radius:
            self.velocity = np.zeros(2)
            self.at_target = True

    def avoid_obstacles_and_robots(self):
        if self.at_target:
            return False

        buffer = 0.1  # Буферное расстояние для избежания столкновений
        num_checks = 1  # Количество проверочных точек вдоль предполагаемой траектории
        best_direction = None
        best_score = float('-inf')
        safe_distance = self.max_speed / num_checks

        # Итерация по разным направлениям в поисках безопасного пути
        angles = np.linspace(-np.pi/2, np.pi/2, 18)
        for angle in angles:
            test_direction = np.array([np.cos(self.angle + angle), np.sin(self.angle + angle)])
            collision = False
            for step in range(1, num_checks + 1):
                test_position = self.position + test_direction * safe_distance * step
                if self.check_collision(test_position, buffer):
                    collision = True
                    break

            if not collision:
                # Оценка направления по его близости к целевому
                score = np.dot(test_direction, self.target - self.position)
                if score > best_score:
                    best_score = score
                    best_direction = test_direction

        # Применение найденного лучшего направления
        if best_direction is not None:
            self.velocity = best_direction * self.max_speed
            return True
        return False

    def check_collision(self, position, buffer):
        """ Проверяет, пересекается ли данная позиция с каким-либо препятствием, учитывая буфер. """
        for obs in self.obstacles:
            expanded_rect = patches.Rectangle((obs['x'] - buffer, obs['y'] - buffer), obs['width'] + 2*buffer, obs['height'] + 2*buffer)
            if expanded_rect.contains_point(position):
                return True
        return False

    def find_alternative_path(self):
        target_direction = (self.target - self.position) / np.linalg.norm(self.target - self.position)
        min_angle = float('inf')
        best_direction = self.velocity
        for angle in np.linspace(-np.pi, np.pi, 36, endpoint=True):
            direction = np.array([np.cos(angle), np.sin(angle)])
            angle_diff = np.arccos(np.clip(np.dot(direction, target_direction), -1, 1))
            if angle_diff < min_angle:
                test_position = self.position + direction * self.max_speed
                collision = False
                for obs in self.obstacles:
                    rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'])
                    if rect.contains_point(test_position):
                        collision = True
                        break
                if not collision:
                    min_angle = angle_diff
                    best_direction = direction
        self.velocity = best_direction * self.max_speed

    def measure_distance(self):
        lidar_direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        for test_point in np.linspace(0, self.lidar_range, 100):
            test_pos = self.position + lidar_direction * test_point
            for obs in self.obstacles:
                rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'])
                if rect.contains_point(test_pos):
                    return test_point
        return self.lidar_range

    def avoid_obstacle(self):
        # Попытка изменить направление на +/- 30, 60, 90 градусов от текущего угла
        for angle_offset in [np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3, np.pi / 2, -np.pi / 2]:
            new_angle = self.angle + angle_offset
            test_direction = np.array([np.cos(new_angle), np.sin(new_angle)])
            test_position = self.position + test_direction * self.max_speed
            if not any(patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height']).contains_point(test_position) for obs in self.obstacles):
                self.angle = new_angle
                self.velocity = test_direction * self.max_speed
                break

    def lidar_visual(self):
        lidar_direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        end_point = self.position + lidar_direction * self.measure_distance()
        return self.position, end_point


class AnimatedPSOSimulation:
    def __init__(self, init_positions, target_positions, obstacles):
        self.robots = [
            PSORobot(x, y, target, obstacles)
            for (x, y), target in zip(init_positions, target_positions)
        ]
        for robot in self.robots:
            robot.update_all_robots(self.robots)

        self.global_best_position = self.find_global_best()
        self.width, self.height = 10, 10
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')  # Сохраняем аспект при обновлении
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.obstacles = obstacles
        self.start_time = time.time()  # Засекаем время старта

    def find_global_best(self):
        best_value = float('inf')
        best_position = None
        for robot in self.robots:
            value = np.linalg.norm(robot.position - robot.target)
            if value < best_value:
                best_value = value
                best_position = robot.position
        return best_position

    def animate(self):
        def update(frame):
            self.ax.clear()
            self.ax.set_aspect('equal')  # Сохраняем аспект при обновлении
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)

            # Draw obstacles
            for obs in self.obstacles:
                rect = patches.Rectangle((obs['x'], obs['y']), obs['width'], obs['height'], linewidth=1, edgecolor='r', facecolor='none')
                self.ax.add_patch(rect)

            # Update global best position
            current_global_best = self.find_global_best()
            if current_global_best is not None:
                self.global_best_position = current_global_best

            all_at_target = True
            for robot in self.robots:
                robot.update_velocity(self.global_best_position)
                robot.move()
                path_x, path_y = zip(*robot.path)
                self.ax.plot(path_x, path_y, 'k-')
                self.ax.plot(robot.position[0], robot.position[1], 'bo')
                self.ax.plot(robot.target[0], robot.target[1], 'rx')

                # Lidar visualization
                start, end = robot.lidar_visual()
                self.ax.plot([start[0], end[0]], [start[1], end[1]], 'g-')

                if not robot.at_target:
                    all_at_target = False

            if all_at_target:
                end_time = time.time()
                print(f"All robots have reached their targets. Total time: {end_time - self.start_time:.2f} seconds.")
                plt.close(self.fig)  # Закрываем фигуру, чтобы остановить анимацию

        anim = FuncAnimation(self.fig, update, frames=np.arange(100), repeat=False)
        plt.show()


# Example setup
init_positions = [(0, 1), (2, 0), (1, 1), (0, 0)]
target_positions = [(8, 7), (7, 7), (8, 8), (8.5, 8)]
obstacles = [{'x': 4, 'y': 4, 'width': 1, 'height': 2}, {'x': 5, 'y': 1, 'width': 2, 'height': 1}, {'x': 6, 'y': 5, 'width': 1.5, 'height': 1.5}]
simulation = AnimatedPSOSimulation(init_positions, target_positions, obstacles)
simulation.animate()
