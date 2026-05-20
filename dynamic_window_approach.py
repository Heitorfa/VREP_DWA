import math

import numpy as np


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DWAController:
    def __init__(self):
        self.max_speed = 0.35
        self.min_speed = 0.0
        self.max_yaw_rate = 1.2
        self.max_accel = 0.7
        self.max_delta_yaw_rate = 2.5

        self.v_resolution = 0.02
        self.w_resolution = 0.08
        self.dt = 0.1
        self.predict_time = 2.5

        self.robot_radius = 0.24
        self.safety_margin = 0.06

        self.to_goal_cost_gain = 0.25
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.45
        self.distance_cost_gain = 2.2

    def calculate_dynamic_window(self, v, w):
        robot_limit = [
            self.min_speed,
            self.max_speed,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ]
        dynamic_limit = [
            v - self.max_accel * self.dt,
            v + self.max_accel * self.dt,
            w - self.max_delta_yaw_rate * self.dt,
            w + self.max_delta_yaw_rate * self.dt,
        ]

        return [
            max(robot_limit[0], dynamic_limit[0]),
            min(robot_limit[1], dynamic_limit[1]),
            max(robot_limit[2], dynamic_limit[2]),
            min(robot_limit[3], dynamic_limit[3]),
        ]

    def motion(self, x, v, w):
        x = np.array(x, dtype=float)
        x[2] = normalize_angle(x[2] + w * self.dt)
        x[0] += v * math.cos(x[2]) * self.dt
        x[1] += v * math.sin(x[2]) * self.dt
        return x

    def predict_trajectory(self, x_init, v, w):
        x = np.array(x_init, dtype=float)
        trajectory = [x.copy()]
        time = 0.0

        while time <= self.predict_time:
            x = self.motion(x, v, w)
            trajectory.append(x.copy())
            time += self.dt

        return np.array(trajectory)

    def calc_to_goal_cost(self, trajectory, goal):
        final = trajectory[-1]
        goal_angle = math.atan2(goal[1] - final[1], goal[0] - final[0])
        return abs(normalize_angle(goal_angle - final[2]))

    def calc_obstacle_cost(self, trajectory, obstacles, v):
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        dx = trajectory[:, 0:1] - obstacles[:, 0]
        dy = trajectory[:, 1:2] - obstacles[:, 1]
        distances = np.hypot(dx, dy)
        min_distance = float(np.min(distances))

        stop_distance = (v * v) / (2.0 * self.max_accel) if self.max_accel > 0 else 0.0
        soft_clearance = self.robot_radius + self.safety_margin + 0.5 * stop_distance

        if min_distance <= self.robot_radius:
            return float("inf")

        cost = 1.0 / (min_distance - self.robot_radius)

        if min_distance < soft_clearance:
            cost += 8.0 * (soft_clearance - min_distance) / soft_clearance

        return cost

    def plan(self, x, v, w, goal, obstacles):
        dynamic_window = self.calculate_dynamic_window(v, w)
        best_u = [0.0, 0.0]
        best_trajectory = self.predict_trajectory(x, 0.0, 0.0)
        min_cost = float("inf")
        current_goal_distance = math.hypot(goal[0] - x[0], goal[1] - x[1])

        for candidate_v in np.arange(
            dynamic_window[0],
            dynamic_window[1] + self.v_resolution,
            self.v_resolution,
        ):
            if candidate_v > dynamic_window[1]:
                continue

            for candidate_w in np.arange(
                dynamic_window[2],
                dynamic_window[3] + self.w_resolution,
                self.w_resolution,
            ):
                if candidate_w > dynamic_window[3]:
                    continue

                trajectory = self.predict_trajectory(x, candidate_v, candidate_w)

                obstacle_cost = self.calc_obstacle_cost(
                    trajectory,
                    obstacles,
                    candidate_v,
                )
                if math.isinf(obstacle_cost):
                    continue

                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(
                    trajectory,
                    goal,
                )
                speed_cost = self.speed_cost_gain * (self.max_speed - candidate_v)
                final_goal_distance = math.hypot(
                    goal[0] - trajectory[-1, 0],
                    goal[1] - trajectory[-1, 1],
                )
                distance_cost = self.distance_cost_gain * final_goal_distance
                progress_reward = 2.0 * max(0.0, current_goal_distance - final_goal_distance)
                total_cost = (
                    to_goal_cost
                    + speed_cost
                    + self.obstacle_cost_gain * obstacle_cost
                    + distance_cost
                    - progress_reward
                )

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_u = [float(candidate_v), float(candidate_w)]
                    best_trajectory = trajectory

        if math.isinf(min_cost):
            goal_angle = math.atan2(goal[1] - x[1], goal[0] - x[0])
            turn = normalize_angle(goal_angle - x[2])
            best_u = [0.08, 0.8 if turn >= 0.0 else -0.8]
            best_trajectory = self.predict_trajectory(x, best_u[0], best_u[1])

        return best_u, best_trajectory


class AStarPlanner:
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def __init__(self, ox, oy, resolution=0.1, rr=0.28):
        self.resolution = resolution
        self.rr = rr
        self.min_x = math.floor(min(ox))
        self.min_y = math.floor(min(oy))
        self.max_x = math.ceil(max(ox))
        self.max_y = math.ceil(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [-1, -1, math.sqrt(2)],
        ]
        self.obstacle_map = [
            [False for _ in range(self.y_width)] for _ in range(self.x_width)
        ]

        inflation = math.ceil(self.rr / self.resolution)

        for iox, ioy in zip(ox, oy):
            center_x = self.calc_xy_index(iox, self.min_x)
            center_y = self.calc_xy_index(ioy, self.min_y)

            for ix in range(center_x - inflation, center_x + inflation + 1):
                for iy in range(center_y - inflation, center_y + inflation + 1):
                    if ix < 0 or iy < 0 or ix >= self.x_width or iy >= self.y_width:
                        continue

                    x = self.calc_grid_position(ix, self.min_x)
                    y = self.calc_grid_position(iy, self.min_y)

                    if math.hypot(iox - x, ioy - y) <= self.rr:
                        self.obstacle_map[ix][iy] = True

    def planning(self, sx, sy, gx, gy):
        start = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
        )
        goal = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
        )
        self.clear_cell(start.x, start.y, radius=2)
        self.clear_cell(goal.x, goal.y, radius=2)

        open_set = {self.calc_grid_index(start): start}
        closed_set = {}

        while open_set:
            current_id = min(
                open_set,
                key=lambda key: open_set[key].cost
                + math.hypot(goal.x - open_set[key].x, goal.y - open_set[key].y),
            )
            current = open_set[current_id]

            if current.x == goal.x and current.y == goal.y:
                goal.parent_index = current.parent_index
                goal.cost = current.cost
                return self.calc_final_path(goal, closed_set)

            del open_set[current_id]
            closed_set[current_id] = current

            for dx, dy, cost in self.motion:
                node = self.Node(current.x + dx, current.y + dy, current.cost + cost, current_id)
                node_id = self.calc_grid_index(node)

                if not self.verify_node(node) or node_id in closed_set:
                    continue

                if node_id not in open_set or open_set[node_id].cost > node.cost:
                    open_set[node_id] = node

        print("A* falhou; usando Goal direto.")
        return [sx, gx], [sy, gy]

    def calc_final_path(self, goal, closed_set):
        rx = [self.calc_grid_position(goal.x, self.min_x)]
        ry = [self.calc_grid_position(goal.y, self.min_y)]
        parent = goal.parent_index

        while parent != -1:
            node = closed_set[parent]
            rx.append(self.calc_grid_position(node.x, self.min_x))
            ry.append(self.calc_grid_position(node.y, self.min_y))
            parent = node.parent_index

        rx.reverse()
        ry.reverse()
        return rx, ry

    def clear_cell(self, cx, cy, radius=1):
        for ix in range(cx - radius, cx + radius + 1):
            for iy in range(cy - radius, cy + radius + 1):
                if 0 <= ix < self.x_width and 0 <= iy < self.y_width:
                    self.obstacle_map[ix][iy] = False

    def verify_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        return not self.obstacle_map[node.x][node.y]

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x
