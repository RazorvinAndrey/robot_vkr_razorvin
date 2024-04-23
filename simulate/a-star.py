import numpy as np


class AStarPathFinder:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.open_list = []
        self.closed_list = set()
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        self.open_list.append((self.f_score[start], start))

    def heuristic(self, node, goal):
        # Эвристика: Манхэттенское расстояние
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def neighbors(self, node):
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-связность
        result = []
        for dir in dirs:
            neighbor = (node[0] + dir[0], node[1] + dir[1])
            if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and self.grid[neighbor] == 0:
                result.append(neighbor)
        return result

    def reconstruct_path(self):
        current = self.goal
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def find_path(self):
        while self.open_list:
            current = min(self.open_list, key=lambda x: x[0])[1]
            if current == self.goal:
                return self.reconstruct_path()

            self.open_list = [item for item in self.open_list if item[1] != current]
            self.closed_list.add(current)

            for neighbor in self.neighbors(current):
                if neighbor in self.closed_list:
                    continue
                tentative_g_score = self.g_score[current] + 1  # Предполагаемая стоимость до соседа

                if neighbor not in [i[1] for i in self.open_list] or tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    if neighbor not in [i[1] for i in self.open_list]:
                        self.open_list.append((self.f_score[neighbor], neighbor))

        return None  # Путь не найден
