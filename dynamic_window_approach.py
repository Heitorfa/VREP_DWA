"""
Dynamic Window Approach (DWA)
Versão otimizada para CoppeliaSim
Com comportamento tipo robô aspirador
"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


show_animation = True


# =========================================================
# TIPOS DE ROBÔ
# =========================================================

class RobotType(Enum):

    circle = 0
    rectangle = 1


# =========================================================
# CONFIGURAÇÕES
# =========================================================

class Config:

    def __init__(self):

        # velocidade linear
        self.max_speed = 0.7
        self.min_speed = -0.2

        # velocidade angular
        self.max_yaw_rate = 180.0 * math.pi / 180.0

        # aceleração
        self.max_accel = 0.8
        self.max_delta_yaw_rate = 180.0 * math.pi / 180.0

        # resolução
        self.v_resolution = 0.02
        self.yaw_rate_resolution = 5.0 * math.pi / 180.0

        # integração
        self.dt = 0.1

        # horizonte de previsão
        self.predict_time = 3.0

        # pesos
        self.to_goal_cost_gain = 2.0
        self.speed_cost_gain = 0.5
        self.obstacle_cost_gain = 1.2

        # anti-travamento
        self.robot_stuck_flag_cons = 0.001

        # tipo do robô
        self.robot_type = RobotType.circle

        # círculo
        self.robot_radius = 0.25

        # retângulo
        self.robot_width = 0.5
        self.robot_length = 1.0


# =========================================================
# CONFIG GLOBAL
# =========================================================

config = Config()


# =========================================================
# DWA PRINCIPAL
# =========================================================

def dwa_control(x, config, goal, ob):

    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(
        x,
        dw,
        config,
        goal,
        ob
    )

    return u, trajectory


# =========================================================
# MODELO DE MOVIMENTO
# =========================================================

def motion(x, u, dt):

    x = np.array(x)

    # orientação
    x[2] += u[1] * dt

    # posição
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    # velocidades
    x[3] = u[0]
    x[4] = u[1]

    return x


# =========================================================
# JANELA DINÂMICA
# =========================================================

def calc_dynamic_window(x, config):

    # limites físicos
    Vs = [

        config.min_speed,
        config.max_speed,

        -config.max_yaw_rate,
        config.max_yaw_rate

    ]

    # limites dinâmicos
    Vd = [

        max(
            config.min_speed,
            x[3] - config.max_accel * config.dt
        ),

        min(
            config.max_speed,
            x[3] + config.max_accel * config.dt
        ),

        x[4] - config.max_delta_yaw_rate * config.dt,

        x[4] + config.max_delta_yaw_rate * config.dt

    ]

    # janela final
    dw = [

        max(Vs[0], Vd[0]),
        min(Vs[1], Vd[1]),

        max(Vs[2], Vd[2]),
        min(Vs[3], Vd[3])

    ]

    return dw


# =========================================================
# PREVISÃO DE TRAJETÓRIA
# =========================================================

def predict_trajectory(x_init, v, y, config):

    x = np.array(x_init)

    # trajetória começa com estado inicial
    trajectory = [x.copy()]

    time = 0.0

    while time <= config.predict_time:

        # evita trajetórias absurdas
        if abs(y) > 1.5:

            break

        x = motion(x, [v, y], config.dt)

        trajectory.append(x.copy())

        time += config.dt

    return np.array(trajectory)


# =========================================================
# ESCOLHA DO MELHOR CONTROLE
# =========================================================

def calc_control_and_trajectory(
        x,
        dw,
        config,
        goal,
        ob):

    x_init = x[:]

    min_cost = float("inf")

    best_u = [0.0, 0.0]

    best_trajectory = np.array([x])

    # =====================================================
    # OBSTÁCULO MAIS PRÓXIMO
    # =====================================================

    nearest_obstacle = 999.0

    if len(ob) > 0:

        dx = ob[:, 0] - x[0]
        dy = ob[:, 1] - x[1]

        nearest_obstacle = np.min(np.hypot(dx, dy))

    # =====================================================
    # PESOS DINÂMICOS
    # =====================================================

    if nearest_obstacle < 0.8:

        # modo evasão
        goal_gain = 0.5
        obstacle_gain = 4.0

    else:

        # modo navegação
        goal_gain = 2.0
        obstacle_gain = 1.0

    # =====================================================
    # TESTA TRAJETÓRIAS
    # =====================================================

    for v in np.arange(
            dw[0],
            dw[1],
            config.v_resolution):

        for y in np.arange(
                dw[2],
                dw[3],
                config.yaw_rate_resolution):

            # =============================================
            # REDUZ VELOCIDADE PERTO DE OBSTÁCULO
            # =============================================

            test_v = v

            if nearest_obstacle < 0.4:

                test_v *= 0.3

            # =============================================
            # ESCAPE DE MÍNIMO LOCAL
            # =============================================

            test_y = y

            if abs(test_v) < 0.05:

                test_y += np.random.uniform(-0.5, 0.5)

            # =============================================
            # PREVISÃO
            # =============================================

            trajectory = predict_trajectory(
                x_init,
                test_v,
                test_y,
                config
            )

            # ignora trajetórias inválidas
            if len(trajectory.shape) == 1:

                continue

            # =============================================
            # CUSTO OBJETIVO
            # =============================================

            to_goal_cost = (
                goal_gain *
                calc_to_goal_cost(
                    trajectory,
                    goal
                )
            )

            # =============================================
            # CUSTO VELOCIDADE
            # =============================================

            speed_cost = (
                config.speed_cost_gain *
                (config.max_speed - trajectory[-1, 3])
            )

            # =============================================
            # CUSTO OBSTÁCULO
            # =============================================

            ob_cost = (
                obstacle_gain *
                calc_obstacle_cost(
                    trajectory,
                    ob,
                    config
                )
            )

            # =============================================
            # PENALIZA GIROS EXAGERADOS
            # =============================================

            heading_cost = abs(test_y) * 0.05

            # =============================================
            # CUSTO FINAL
            # =============================================

            final_cost = (
                to_goal_cost +
                speed_cost +
                ob_cost +
                heading_cost
            )

            # =============================================
            # EVITA GIRAR PARADO
            # =============================================

            if abs(test_v) < 0.05 and abs(test_y) > 0.5:

                final_cost += 5.0

            # =============================================
            # MELHOR TRAJETÓRIA
            # =============================================

            if final_cost < min_cost:

                min_cost = final_cost

                best_u = [test_v, test_y]

                best_trajectory = trajectory

                # anti-travamento
                if (
                    abs(best_u[0]) < config.robot_stuck_flag_cons
                    and
                    abs(x[3]) < config.robot_stuck_flag_cons
                ):

                    best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory


# =========================================================
# CUSTO DE OBSTÁCULO
# =========================================================

def calc_obstacle_cost(
        trajectory,
        ob,
        config):

    if len(ob) == 0:

        return 0.0

    ox = ob[:, 0]
    oy = ob[:, 1]

    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]

    r = np.hypot(dx, dy)

    # colisão
    if np.array(r <= config.robot_radius).any():

        return float("Inf")

    # menor distância
    min_r = np.min(r)

    # custo suavizado
    return 1.0 / (min_r + 0.2)


# =========================================================
# CUSTO PARA OBJETIVO
# =========================================================

def calc_to_goal_cost(
        trajectory,
        goal):

    # proteção
    if len(trajectory.shape) == 1:

        return float("Inf")

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    error_angle = math.atan2(dy, dx)

    cost_angle = error_angle - trajectory[-1, 2]

    cost = abs(
        math.atan2(
            math.sin(cost_angle),
            math.cos(cost_angle)
        )
    )

    return cost


# =========================================================
# DESENHA SETA
# =========================================================

def plot_arrow(
        x,
        y,
        yaw,
        length=0.5,
        width=0.1):

    plt.arrow(
        x,
        y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        head_length=width,
        head_width=width
    )

    plt.plot(x, y)


# =========================================================
# DESENHA ROBÔ
# =========================================================

def plot_robot(
        x,
        y,
        yaw,
        config):

    if config.robot_type == RobotType.circle:

        circle = plt.Circle(
            (x, y),
            config.robot_radius,
            color="b",
            fill=False
        )

        plt.gca().add_artist(circle)

        out_x = x + math.cos(yaw) * config.robot_radius
        out_y = y + math.sin(yaw) * config.robot_radius

        plt.plot([x, out_x], [y, out_y], "-k")

    else:

        outline = np.array([

            [-config.robot_length / 2,
             config.robot_length / 2,
             config.robot_length / 2,
             -config.robot_length / 2,
             -config.robot_length / 2],

            [config.robot_width / 2,
             config.robot_width / 2,
             -config.robot_width / 2,
             -config.robot_width / 2,
             config.robot_width / 2]

        ])

        Rot1 = np.array([

            [math.cos(yaw), math.sin(yaw)],
            [-math.sin(yaw), math.cos(yaw)]

        ])

        outline = (outline.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y

        plt.plot(
            np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(),
            "-k"
        )