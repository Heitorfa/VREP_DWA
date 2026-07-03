"""
Algoritmos de planejamento de caminho para navegacao autonoma.

Este modulo contem:

  * DWAController  -> controlador reativo local (Dynamic Window Approach).
  * AStarPlanner   -> planejador global A* MELHORADO conforme:

        Guo, H. et al. (2024). "Path planning of greenhouse electric crawler
        tractor based on the improved A* and DWA algorithms".
        Computers and Electronics in Agriculture, 227, 109596.

    As tres melhorias do artigo foram implementadas:

      1. Heuristica ponderada:  f(n) = g(n) + (1 + d/D) * h(n)
         onde d = distancia do no atual ao objetivo e D = distancia do
         inicio ao objetivo. O peso varia de ~2 (perto do inicio, busca
         rapida) a ~1 (perto do objetivo, caminho otimo).

      2. Selecao de pontos-chave (key point selection):
         (a) remocao de nos colineares redundantes;
         (b) simplificacao por linha de visao (line-of-sight): se o segmento
             entre o ponto anterior e o posterior a uma curva estiver livre
             de obstaculos, a curva e descartada.

      3. Suavizacao por curva de Bezier de 2a ordem:
             B(t) = (1-t)^2 P0 + 2 t (1-t) P1 + t^2 P2 ,  t em [0, 1]

    O metodo planning() devolve tanto o caminho denso suavizado (para
    rastreamento com lookahead) quanto os pontos-chave (usados como metas
    intermediarias do DWA na fusao dos dois algoritmos).
"""

import math

import numpy as np


def normalize_angle(angle):
    """Normaliza um angulo para o intervalo [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ==========================================================================
#  CONTROLADOR LOCAL - DYNAMIC WINDOW APPROACH (DWA)
# ==========================================================================
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

        # robot_radius -> folga desejada (custo suave, mantem distancia).
        # collision_radius -> raio de colisao REAL (menor); so abaixo dele a
        # trajetoria e descartada. Isso permite aproximar-se de paredes quando
        # o objetivo esta perto delas, sem congelar o robo.
        self.robot_radius = 0.24
        self.collision_radius = 0.16
        self.safety_margin = 0.06

        # Ganhos da funcao de custo. O termo "distance" (distancia final ao
        # alvo local) corresponde ao termo key(v, w) do DWA melhorado do
        # artigo: guia o robo ate o ponto-chave corrente do caminho global.
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

        # Colisao real apenas abaixo do raio de colisao (menor que a folga).
        if min_distance <= self.collision_radius:
            return float("inf")

        cost = 1.0 / (min_distance - self.collision_radius)

        # Folga suave: penaliza (mas NAO proibe) aproximar-se de obstaculos,
        # permitindo atravessar corredores estreitos e chegar a alvos junto
        # a paredes.
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
                # Termo key(v, w) do artigo: distancia euclidiana do fim da
                # trajetoria simulada ao alvo local (ponto-chave corrente).
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


# ==========================================================================
#  PLANEJADOR GLOBAL - A* MELHORADO (Guo et al., 2024)
# ==========================================================================
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

        self.inflation = math.ceil(self.rr / self.resolution)
        self.last_plan_failed = False

        for iox, ioy in zip(ox, oy):
            center_x = self.calc_xy_index(iox, self.min_x)
            center_y = self.calc_xy_index(ioy, self.min_y)

            for ix in range(center_x - self.inflation, center_x + self.inflation + 1):
                for iy in range(center_y - self.inflation, center_y + self.inflation + 1):
                    if ix < 0 or iy < 0 or ix >= self.x_width or iy >= self.y_width:
                        continue

                    x = self.calc_grid_position(ix, self.min_x)
                    y = self.calc_grid_position(iy, self.min_y)

                    if math.hypot(iox - x, ioy - y) <= self.rr:
                        self.obstacle_map[ix][iy] = True

    # ----------------------------------------------------------------------
    #  Busca A* com heuristica ponderada (melhoria 1 do artigo)
    # ----------------------------------------------------------------------
    def planning(self, sx, sy, gx, gy):
        """Planeja o caminho de (sx, sy) ate (gx, gy).

        Retorna quatro listas:
            rx, ry      -> caminho denso suavizado por Bezier (rastreamento)
            kx, ky      -> pontos-chave (metas intermediarias para o DWA)
        Mantida a compatibilidade: os dois primeiros valores continuam sendo
        o caminho (x, y). Os pontos-chave sao um retorno adicional.
        """
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
        # Limpa uma vizinhanca do inicio e do objetivo proporcional a
        # inflacao, garantindo que ambos fiquem sempre acessiveis (crucial
        # quando o objetivo esta encostado numa parede).
        self.last_plan_failed = False
        limpeza = self.inflation + 1
        self.clear_cell(start.x, start.y, radius=limpeza)
        self.clear_cell(goal.x, goal.y, radius=limpeza)

        # D = distancia do inicio ao objetivo (constante), usada no peso.
        start_to_goal = math.hypot(goal.x - start.x, goal.y - start.y)
        start_to_goal = max(start_to_goal, 1e-6)

        open_set = {self.calc_grid_index(start): start}
        closed_set = {}

        while open_set:
            # f(n) = g(n) + (1 + d/D) * h(n), com h e d = distancia
            # euclidiana (em celulas) do no ao objetivo.
            def evaluation(key):
                node = open_set[key]
                d = math.hypot(goal.x - node.x, goal.y - node.y)
                weight = 1.0 + d / start_to_goal
                return node.cost + weight * d

            current_id = min(open_set, key=evaluation)
            current = open_set[current_id]

            if current.x == goal.x and current.y == goal.y:
                goal.parent_index = current.parent_index
                goal.cost = current.cost
                rx, ry = self.calc_final_path(goal, closed_set)
                return self.post_process(rx, ry)

            del open_set[current_id]
            closed_set[current_id] = current

            for dx, dy, cost in self.motion:
                node = self.Node(current.x + dx, current.y + dy, current.cost + cost, current_id)
                node_id = self.calc_grid_index(node)

                if not self.verify_node(node) or node_id in closed_set:
                    continue

                if node_id not in open_set or open_set[node_id].cost > node.cost:
                    open_set[node_id] = node

        # Nenhuma rota encontrada com esta margem de inflacao.
        self.last_plan_failed = True
        straight_x, straight_y = [sx, gx], [sy, gy]
        return self.post_process(straight_x, straight_y)

    # ----------------------------------------------------------------------
    #  Pos-processamento: pontos-chave + suavizacao (melhorias 2 e 3)
    # ----------------------------------------------------------------------
    def post_process(self, rx, ry):
        """Aplica selecao de pontos-chave e suavizacao Bezier.

        Recebe o caminho bruto (centros de celula) e retorna:
            sx, sy  -> caminho denso suavizado
            kx, ky  -> pontos-chave (para o DWA)
        """
        path = list(zip(rx, ry))

        if len(path) <= 2:
            kx = [p[0] for p in path]
            ky = [p[1] for p in path]
            return rx, ry, kx, ky

        # Melhoria 2a: remove nos colineares redundantes.
        key_points = self._remove_collinear(path)

        # Melhoria 2b: simplificacao por linha de visao.
        key_points = self._line_of_sight_simplify(key_points)

        # Melhoria 3: suavizacao por curva de Bezier de 2a ordem.
        smooth = self._bezier_smooth(key_points)

        sx = [p[0] for p in smooth]
        sy = [p[1] for p in smooth]
        kx = [p[0] for p in key_points]
        ky = [p[1] for p in key_points]
        return sx, sy, kx, ky

    @staticmethod
    def _remove_collinear(path, eps=1e-6):
        """Mantem apenas os pontos de curva (remove colineares)."""
        if len(path) <= 2:
            return list(path)

        result = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = result[-1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            # Produto vetorial ~ 0 => tres pontos colineares.
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(cross) > eps:
                result.append(path[i])
        result.append(path[-1])
        return result

    def _line_of_sight_simplify(self, points):
        """Remove curvas cujo 'atalho' (ponto anterior->posterior) e livre."""
        if len(points) <= 2:
            return list(points)

        result = [points[0]]
        i = 1
        while i < len(points) - 1:
            prev_point = result[-1]
            next_point = points[i + 1]
            # Se a linha reta entre o anterior e o proximo estiver livre, a
            # curva atual e redundante e pode ser eliminada.
            if self.is_line_free(prev_point, next_point):
                i += 1  # pula o ponto atual (nao adiciona)
            else:
                result.append(points[i])
                i += 1
        result.append(points[-1])
        return result

    def is_line_free(self, p0, p1):
        """Verifica se o segmento p0->p1 esta livre de obstaculos.

        Amostra o segmento em passos de ~meia celula e consulta o mapa de
        ocupacao inflado ja construido no __init__.
        """
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(length / (self.resolution * 0.5)) + 1)

        for s in range(steps + 1):
            t = s / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            ix = self.calc_xy_index(x, self.min_x)
            iy = self.calc_xy_index(y, self.min_y)
            if ix < 0 or iy < 0 or ix >= self.x_width or iy >= self.y_width:
                return False
            if self.obstacle_map[ix][iy]:
                return False
        return True

    @staticmethod
    def _bezier_smooth(points, samples=12, corner_ratio=0.35):
        """Suaviza cada curva com uma Bezier de 2a ordem.

        Para cada ponto interno P1 (curva), constroi-se uma Bezier
        quadratica cujos extremos ficam sobre os segmentos vizinhos:
            P0' = P1 + corner_ratio*(P_ant - P1)
            P2' = P1 + corner_ratio*(P_prox - P1)
            B(t) = (1-t)^2 P0' + 2 t (1-t) P1 + t^2 P2'
        """
        if len(points) <= 2:
            return list(points)

        pts = [np.array(p, dtype=float) for p in points]
        smooth = [tuple(pts[0])]

        for i in range(1, len(pts) - 1):
            p_prev = pts[i - 1]
            p_cur = pts[i]
            p_next = pts[i + 1]

            p0 = p_cur + corner_ratio * (p_prev - p_cur)
            p2 = p_cur + corner_ratio * (p_next - p_cur)

            # Reta de entrada ate o inicio do arco.
            smooth.append(tuple(p0))
            for s in range(1, samples):
                t = s / samples
                b = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p_cur + t ** 2 * p2
                smooth.append(tuple(b))
            smooth.append(tuple(p2))

        smooth.append(tuple(pts[-1]))
        return smooth

    # ----------------------------------------------------------------------
    #  Utilitarios da grade
    # ----------------------------------------------------------------------
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
