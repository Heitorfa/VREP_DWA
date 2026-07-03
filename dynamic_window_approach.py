"""
Algoritmos de navegacao: A* MELHORADO (global) + DWA (local).

Referencias:
  * Guo, H. et al. (2024). "Path planning of greenhouse electric crawler
    tractor based on the improved A* and DWA algorithms". Computers and
    Electronics in Agriculture, 227, 109596.
      - Heuristica ponderada: f(n) = g(n) + (1 + d/D) * h(n)
      - Selecao de pontos-chave (colineares + linha de visao)
      - Suavizacao por curva de Bezier de 2a ordem
      - Fusao: pontos-chave do A* como metas intermediarias do DWA
  * Fox, Burgard & Thrun (1997). "The Dynamic Window Approach to
    Collision Avoidance".

Robustez adicional (nao presente no artigo, necessaria na pratica):
  * Penalidade de proximidade no A* (estilo costmap): prefere corredores
    largos e so usa passagens apertadas quando nao ha alternativa.
  * Custo de obstaculo LIMITADO no DWA (1/d): evita que o robo congele na
    entrada de corredores estreitos.
  * Margem de inflacao adaptativa (planejar_rota): reduz a margem apenas
    se nao existir rota com a margem conservadora.
"""

import heapq
import math

import numpy as np


def normalize_angle(angle):
    """Normaliza um angulo para o intervalo [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ==========================================================================
#  CONTROLADOR LOCAL - DYNAMIC WINDOW APPROACH (DWA)
# ==========================================================================
class DWAController:
    """Amostra velocidades (v, w) na janela dinamica e escolhe a trajetoria
    simulada de menor custo. O termo de distancia ao alvo local corresponde
    ao key(v, w) do DWA melhorado do artigo."""

    def __init__(self):
        # Limites cinematicos e dinamicos
        self.max_speed = 0.35            # m/s
        self.min_speed = 0.0
        self.max_yaw_rate = 1.2          # rad/s
        self.max_accel = 0.7             # m/s^2
        self.max_delta_yaw_rate = 2.5    # rad/s^2

        # Amostragem e horizonte de simulacao
        self.v_resolution = 0.02
        self.w_resolution = 0.08
        self.dt = 0.1
        self.predict_time = 2.5

        # ------------------------------------------------------------------
        # RAIOS DE SEGURANCA (ajuste principal para passagens apertadas)
        #   collision_radius -> raio FISICO do robo; abaixo disso = colisao.
        #   robot_radius     -> folga PREFERIDA ("distancia aceitavel"):
        #                       abaixo dela apenas penaliza, nunca proibe.
        # Menor corredor transponivel ~ 2*collision_radius + ~0.10 m.
        # ------------------------------------------------------------------
        self.robot_radius = 0.18
        self.collision_radius = 0.12
        self.safety_margin = 0.04

        # Ganhos da funcao de custo
        self.to_goal_cost_gain = 0.25    # alinhamento com o alvo
        self.speed_cost_gain = 1.0       # preferencia por andar rapido
        self.obstacle_cost_gain = 0.45   # afastamento de obstaculos
        self.distance_cost_gain = 2.2    # termo key(v,w): distancia ao alvo

    # ----------------------------------------------------------------------
    def calculate_dynamic_window(self, v, w):
        """Intersecao dos limites absolutos com os alcancaveis em dt."""
        return [
            max(self.min_speed, v - self.max_accel * self.dt),
            min(self.max_speed, v + self.max_accel * self.dt),
            max(-self.max_yaw_rate, w - self.max_delta_yaw_rate * self.dt),
            min(self.max_yaw_rate, w + self.max_delta_yaw_rate * self.dt),
        ]

    def predict_trajectory(self, x_init, v, w):
        """Simula a trajetoria de (v, w) constante por predict_time."""
        steps = int(self.predict_time / self.dt) + 1
        traj = np.empty((steps + 1, 3))
        x, y, th = float(x_init[0]), float(x_init[1]), float(x_init[2])
        traj[0] = (x, y, th)

        for i in range(1, steps + 1):
            th = normalize_angle(th + w * self.dt)
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            traj[i] = (x, y, th)

        return traj

    # ----------------------------------------------------------------------
    def calc_to_goal_cost(self, trajectory, goal):
        """Desalinhamento angular do fim da trajetoria com o alvo."""
        fx, fy, fth = trajectory[-1]
        goal_angle = math.atan2(goal[1] - fy, goal[0] - fx)
        return abs(normalize_angle(goal_angle - fth))

    def calc_obstacle_cost(self, trajectory, obstacles, v):
        """Custo de proximidade LIMITADO (1/d).

        Um custo que explode perto do raio faz "ficar parado" custar menos
        que atravessar um corredor (duas paredes proximas ao mesmo tempo) e
        congela o robo na entrada. Com 1/d limitado, o avanco volta a
        dominar a decisao; a colisao real continua proibida.
        """
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        dx = trajectory[:, 0:1] - obstacles[:, 0]
        dy = trajectory[:, 1:2] - obstacles[:, 1]
        min_distance = float(np.min(np.hypot(dx, dy)))

        if min_distance <= self.collision_radius:
            return float("inf")

        cost = 1.0 / min_distance

        stop = (v * v) / (2.0 * self.max_accel)
        soft = self.robot_radius + self.safety_margin + 0.5 * stop
        if min_distance < soft:
            cost += 2.0 * (soft - min_distance) / soft

        return cost

    # ----------------------------------------------------------------------
    def plan(self, x, v, w, goal, obstacles):
        """Escolhe o melhor comando [v, w] para o estado x = (x, y, theta)."""
        vmin, vmax, wmin, wmax = self.calculate_dynamic_window(v, w)
        best_u = [0.0, 0.0]
        best_trajectory = self.predict_trajectory(x, 0.0, 0.0)
        min_cost = float("inf")
        goal_dist_now = math.hypot(goal[0] - x[0], goal[1] - x[1])

        for cv in np.arange(vmin, vmax + self.v_resolution / 2, self.v_resolution):
            for cw in np.arange(wmin, wmax + self.w_resolution / 2, self.w_resolution):
                traj = self.predict_trajectory(x, cv, cw)

                obstacle_cost = self.calc_obstacle_cost(traj, obstacles, cv)
                if math.isinf(obstacle_cost):
                    continue

                goal_dist_end = math.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])

                total = (
                    self.to_goal_cost_gain * self.calc_to_goal_cost(traj, goal)
                    + self.speed_cost_gain * (self.max_speed - cv)
                    + self.obstacle_cost_gain * obstacle_cost
                    + self.distance_cost_gain * goal_dist_end
                    - 2.0 * max(0.0, goal_dist_now - goal_dist_end)
                )

                if total < min_cost:
                    min_cost = total
                    best_u = [float(cv), float(cw)]
                    best_trajectory = traj

        if math.isinf(min_cost):
            # Todas as trajetorias colidem: gira no eixo em direcao ao alvo.
            turn = normalize_angle(math.atan2(goal[1] - x[1], goal[0] - x[0]) - x[2])
            best_u = [0.05, 0.8 if turn >= 0.0 else -0.8]
            best_trajectory = self.predict_trajectory(x, best_u[0], best_u[1])

        return best_u, best_trajectory


# ==========================================================================
#  PLANEJADOR GLOBAL - A* MELHORADO (Guo et al., 2024) + costmap
# ==========================================================================
class AStarPlanner:
    """A* em grade com heuristica ponderada, penalidade de proximidade,
    selecao de pontos-chave e suavizacao de Bezier."""

    # Penalidade de proximidade (estilo costmap do ROS)
    CLEARANCE_CELLS = 8       # alcance da penalidade (celulas)
    CLEARANCE_WEIGHT = 5.0    # peso maximo por celula percorrida

    MOTION = [
        (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def __init__(self, pontos_obstaculo, resolution=0.1, rr=0.20):
        """pontos_obstaculo: array Nx2 de pontos (x, y) no mundo."""
        pontos = np.asarray(pontos_obstaculo, dtype=float).reshape(-1, 2)
        self.resolution = resolution
        self.rr = rr
        self.last_plan_failed = False

        self.min_x = math.floor(float(pontos[:, 0].min()))
        self.min_y = math.floor(float(pontos[:, 1].min()))
        self.x_width = round((math.ceil(float(pontos[:, 0].max())) - self.min_x) / resolution)
        self.y_width = round((math.ceil(float(pontos[:, 1].max())) - self.min_y) / resolution)

        self._build_obstacle_map(pontos)
        self._build_clearance_penalty()

    # ------------------------------------------------------------------
    def _build_obstacle_map(self, pontos):
        """Marca celulas ocupadas inflando cada ponto pelo raio rr."""
        self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool)
        inflation = math.ceil(self.rr / self.resolution)

        ix = np.round((pontos[:, 0] - self.min_x) / self.resolution).astype(int)
        iy = np.round((pontos[:, 1] - self.min_y) / self.resolution).astype(int)

        for cx, cy, (px, py) in zip(ix, iy, pontos):
            x0, x1 = max(cx - inflation, 0), min(cx + inflation + 1, self.x_width)
            y0, y1 = max(cy - inflation, 0), min(cy + inflation + 1, self.y_width)
            if x0 >= x1 or y0 >= y1:
                continue
            gx = self.min_x + np.arange(x0, x1) * self.resolution
            gy = self.min_y + np.arange(y0, y1) * self.resolution
            mask = np.hypot(gx[:, None] - px, gy[None, :] - py) <= self.rr
            self.obstacle_map[x0:x1, y0:y1] |= mask

    def _build_clearance_penalty(self):
        """Distancia (aproximada) de cada celula livre ao obstaculo mais
        proximo, por dilatacao iterativa; penalidade decai com o quadrado."""
        dist = np.full(self.obstacle_map.shape, self.CLEARANCE_CELLS, dtype=float)
        dist[self.obstacle_map] = 0.0

        frente = self.obstacle_map.copy()
        for k in range(1, self.CLEARANCE_CELLS):
            d = frente.copy()
            d[1:, :] |= frente[:-1, :]
            d[:-1, :] |= frente[1:, :]
            d[:, 1:] |= frente[:, :-1]
            d[:, :-1] |= frente[:, 1:]
            d[1:, 1:] |= frente[:-1, :-1]
            d[1:, :-1] |= frente[:-1, 1:]
            d[:-1, 1:] |= frente[1:, :-1]
            d[:-1, :-1] |= frente[1:, 1:]
            dist[d & ~frente] = float(k)
            frente = d

        ratio = (self.CLEARANCE_CELLS - dist) / self.CLEARANCE_CELLS
        self.penalty_map = self.CLEARANCE_WEIGHT * ratio * ratio

    # ------------------------------------------------------------------
    #  Conversoes grade <-> mundo
    # ------------------------------------------------------------------
    def to_index(self, pos, min_pos):
        return round((pos - min_pos) / self.resolution)

    def to_world(self, index, min_pos):
        return index * self.resolution + min_pos

    def dentro(self, ix, iy):
        return 0 <= ix < self.x_width and 0 <= iy < self.y_width

    def clear_cell(self, cx, cy, radius):
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, self.x_width)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, self.y_width)
        self.obstacle_map[x0:x1, y0:y1] = False

    # ------------------------------------------------------------------
    #  Busca A* (heuristica ponderada + penalidade de proximidade)
    # ------------------------------------------------------------------
    def planning(self, sx, sy, gx, gy):
        """Planeja de (sx, sy) ate (gx, gy).

        Retorna (caminho, pontos_chave): listas de tuplas (x, y) no mundo.
        caminho e denso e suavizado (rastreamento); pontos_chave sao as
        metas intermediarias para o DWA. Em falha, last_plan_failed = True
        e o caminho devolvido e a linha reta inicio->objetivo.
        """
        self.last_plan_failed = False

        start = (self.to_index(sx, self.min_x), self.to_index(sy, self.min_y))
        goal = (self.to_index(gx, self.min_x), self.to_index(gy, self.min_y))

        # Inicio e objetivo sempre acessiveis (objetivo junto a parede etc.)
        limpeza = math.ceil(self.rr / self.resolution) + 1
        self.clear_cell(*start, radius=limpeza)
        self.clear_cell(*goal, radius=limpeza)

        # D (dist. inicio->objetivo) para o peso adaptativo da heuristica.
        D = max(math.hypot(goal[0] - start[0], goal[1] - start[1]), 1e-6)

        def f(celula, g_cost):
            d = math.hypot(goal[0] - celula[0], goal[1] - celula[1])
            return g_cost + (1.0 + d / D) * d

        g_score = {start: 0.0}
        parent = {start: None}
        open_heap = [(f(start, 0.0), start)]
        closed = set()

        while open_heap:
            _, cell = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)

            if cell == goal:
                return self._post_process(self._reconstruct(parent, goal))

            cx, cy = cell
            for dx, dy, move in self.MOTION:
                nx_, ny_ = cx + dx, cy + dy
                if not self.dentro(nx_, ny_) or self.obstacle_map[nx_, ny_]:
                    continue
                vizinho = (nx_, ny_)
                if vizinho in closed:
                    continue

                # custo do passo + penalidade de proximidade da celula:
                # corredores largos ficam mais baratos que atalhos raspando
                # em cantos/vaos apertados.
                g_new = g_score[cell] + move + float(self.penalty_map[nx_, ny_])
                if g_new < g_score.get(vizinho, float("inf")):
                    g_score[vizinho] = g_new
                    parent[vizinho] = cell
                    heapq.heappush(open_heap, (f(vizinho, g_new), vizinho))

        self.last_plan_failed = True
        return self._post_process([(sx, sy), (gx, gy)])

    def _reconstruct(self, parent, goal):
        cells = []
        cell = goal
        while cell is not None:
            cells.append(cell)
            cell = parent[cell]
        cells.reverse()
        return [
            (self.to_world(cx, self.min_x), self.to_world(cy, self.min_y))
            for cx, cy in cells
        ]

    # ------------------------------------------------------------------
    #  Pos-processamento (melhorias 2 e 3 do artigo)
    # ------------------------------------------------------------------
    def _post_process(self, caminho):
        if len(caminho) <= 2:
            return list(caminho), list(caminho)

        pontos_chave = self._remove_collinear(caminho)
        pontos_chave = self._line_of_sight_simplify(pontos_chave)
        suave = self._bezier_smooth(pontos_chave)
        return suave, pontos_chave

    @staticmethod
    def _remove_collinear(path, eps=1e-6):
        """Mantem apenas os pontos de curva (remove colineares)."""
        result = [path[0]]
        for i in range(1, len(path) - 1):
            (ax, ay), (bx, by), (cx, cy) = result[-1], path[i], path[i + 1]
            if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > eps:
                result.append(path[i])
        result.append(path[-1])
        return result

    def _line_of_sight_simplify(self, points):
        """Remove curvas cujo atalho (anterior->posterior) e livre E mantem
        folga razoavel (não corta de volta por vaos apertados)."""
        if len(points) <= 2:
            return list(points)

        ratio = (self.CLEARANCE_CELLS - 2.0) / self.CLEARANCE_CELLS
        limite = self.CLEARANCE_WEIGHT * ratio * ratio

        result = [points[0]]
        for i in range(1, len(points) - 1):
            if not self.is_line_free(result[-1], points[i + 1], max_penalty=limite):
                result.append(points[i])
        result.append(points[-1])
        return result

    def is_line_free(self, p0, p1, max_penalty=None):
        """Segmento p0->p1 livre de obstaculos (e, opcionalmente, de celulas
        com folga baixa)."""
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(2, int(length / (self.resolution * 0.5)) + 1)

        for s in range(steps + 1):
            t = s / steps
            ix = self.to_index(p0[0] + (p1[0] - p0[0]) * t, self.min_x)
            iy = self.to_index(p0[1] + (p1[1] - p0[1]) * t, self.min_y)
            if not self.dentro(ix, iy) or self.obstacle_map[ix, iy]:
                return False
            if max_penalty is not None and float(self.penalty_map[ix, iy]) > max_penalty:
                return False
        return True

    @staticmethod
    def _bezier_smooth(points, samples=12, corner_ratio=0.35):
        """Substitui cada curva por um arco de Bezier de 2a ordem:
        B(t) = (1-t)^2 P0' + 2t(1-t) P1 + t^2 P2'."""
        if len(points) <= 2:
            return list(points)

        pts = [np.array(p, dtype=float) for p in points]
        smooth = [tuple(pts[0])]

        for i in range(1, len(pts) - 1):
            p0 = pts[i] + corner_ratio * (pts[i - 1] - pts[i])
            p2 = pts[i] + corner_ratio * (pts[i + 1] - pts[i])
            smooth.append(tuple(p0))
            for s in range(1, samples):
                t = s / samples
                b = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * pts[i] + t ** 2 * p2
                smooth.append(tuple(b))
            smooth.append(tuple(p2))

        smooth.append(tuple(pts[-1]))
        return smooth


# ==========================================================================
#  PLANEJAMENTO COM MARGEM ADAPTATIVA
# ==========================================================================
def planejar_rota(pontos_obstaculo, inicio, objetivo,
                  resolution=0.1, margens=(0.20, 0.16, 0.12)):
    """Planeja com margens de inflacao decrescentes.

    Tenta a margem conservadora primeiro; so afrouxa se nao houver rota
    (corredor estreito, objetivo junto a parede). A penalidade de
    proximidade garante que, mesmo com margem pequena, corredores largos
    sejam preferidos a vaos apertados.

    Retorna (caminho, pontos_chave, planner, rr_usado). Se nenhuma margem
    encontrar rota, devolve a linha reta e planner.last_plan_failed = True.
    """
    planner = None
    for rr in margens:
        planner = AStarPlanner(pontos_obstaculo, resolution=resolution, rr=rr)
        caminho, pontos_chave = planner.planning(
            inicio[0], inicio[1], objetivo[0], objetivo[1]
        )
        if not planner.last_plan_failed:
            return caminho, pontos_chave, planner, rr

    return caminho, pontos_chave, planner, margens[-1]
