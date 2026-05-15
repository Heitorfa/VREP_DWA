from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import dynamic_window_approach as dw

import matplotlib.pyplot as plt
import numpy as np
import math


# =========================================================
# CONEXÃO COM COPPELIASIM
# =========================================================

client = RemoteAPIClient()

sim = client.require('sim')


# =========================================================
# CONFIGURAÇÕES DOS SENSORES
# =========================================================

sensor_pos = [

    (0.15, 0.00),

    (0.12, -0.10),
    (0.12,  0.10),

    (0.00, -0.15),
    (0.00,  0.15)

]

sensor_ori = [

    0,
    -45,
    45,
    -90,
    90

]

sensor_ori = [math.radians(a) for a in sensor_ori]


# =========================================================
# VARIÁVEIS GLOBAIS
# =========================================================

param = {}

config = dw.Config()


# =========================================================
# DETECTA OBSTÁCULOS
# =========================================================

def get_ob(x):

    ob = []

    for i, sensor in enumerate(param['sensors']):

        result, distance, point, obj, normal = sim.readProximitySensor(sensor)

        if result > 0:

            d = math.sqrt(
                point[0]**2 +
                point[1]**2 +
                point[2]**2
            )

            # posição do sensor no mundo
            sx = sensor_pos[i][0]
            sy = sensor_pos[i][1]

            gx = (
                x[0] +
                sx * math.cos(x[2]) -
                sy * math.sin(x[2])
            )

            gy = (
                x[1] +
                sx * math.sin(x[2]) +
                sy * math.cos(x[2])
            )

            # direção do sensor
            angle = x[2] + sensor_ori[i]

            # posição estimada do obstáculo
            ox = gx + d * math.cos(angle)

            oy = gy + d * math.sin(angle)

            ob.append([ox, oy])

    if len(ob) == 0:

        return np.array([[-100, -100]])

    return np.array(ob)


# =========================================================
# MOVIMENTO DO ROBÔ
# =========================================================

def rap_motion(u):

    # =====================================================
    # CONVERSÃO DWA -> RODAS
    # =====================================================

    wr = (u[0] + u[1] * config.c) / config.r

    wl = (u[0] - u[1] * config.c) / config.r

    # =====================================================
    # LIMITADOR
    # =====================================================

    wr = max(min(wr, 8), -8)

    wl = max(min(wl, 8), -8)

    # =====================================================
    # ENVIA PARA AS RODAS
    # =====================================================

    # IMPORTANTE:
    # se girar errado:
    # troque sinais aqui

    sim.setJointTargetVelocity(
        param['motorRight'],
        wr
    )

    sim.setJointTargetVelocity(
        param['motorLeft'],
        wl
    )

    # executa física
    sim.step()

    # =====================================================
    # LÊ POSIÇÃO
    # =====================================================

    position = sim.getObjectPosition(
        param['robot'],
        -1
    )

    orientation = sim.getObjectOrientation(
        param['robot'],
        -1
    )

    x = np.array([

        position[0],
        position[1],
        orientation[2],

        u[0],
        u[1]

    ])

    return x


# =========================================================
# INICIALIZAÇÃO
# =========================================================

def init():

    # =====================================================
    # ROBÔ
    # =====================================================

    param['robot'] = sim.getObject('/Cuboid')

    # =====================================================
    # MOTORES
    # =====================================================

    param['motorRight'] = sim.getObject('/MOTOR_DIREITO')

    param['motorLeft'] = sim.getObject('/MOTOR_ESQUERDO')

    # =====================================================
    # SENSORES
    # =====================================================

    param['sensors'] = [

        sim.getObject('/SENSOR_MEIO'),

        sim.getObject('/SENSOR_DIAG_DIREITO'),

        sim.getObject('/SENSOR_DIAG_ESQUERDO'),

        sim.getObject('/SENSOR_DIREITO'),

        sim.getObject('/SENSOR_ESQUERDO')

    ]

    # =====================================================
    # POSIÇÃO INICIAL
    # =====================================================

    position = sim.getObjectPosition(
        param['robot'],
        -1
    )

    orientation = sim.getObjectOrientation(
        param['robot'],
        -1
    )

    x = np.array([

        position[0],
        position[1],
        orientation[2],

        0.0,
        0.0

    ])

    # =====================================================
    # OBJETIVO
    # =====================================================

    config.goal = np.array([2.5, 2.5])

    # =====================================================
    # CONFIGURAÇÕES DWA
    # =====================================================

    config.robot_type = dw.RobotType.circle

    config.max_speed = 0.7

    config.min_speed = -0.2

    config.max_yaw_rate = 180.0 * math.pi / 180.0

    config.max_accel = 0.8

    config.max_delta_yaw_rate = 180.0 * math.pi / 180.0

    config.v_resolution = 0.02

    config.yaw_rate_resolution = 5.0 * math.pi / 180.0

    config.dt = 0.1

    config.predict_time = 1.5

    config.to_goal_cost_gain = 1.0

    config.speed_cost_gain = 0.3

    config.obstacle_cost_gain = 3.0

    config.robot_radius = 0.25

    # =====================================================
    # CINEMÁTICA DIFERENCIAL
    # =====================================================

    # distância entre rodas
    config.c = 0.18

    # raio da roda
    config.r = 0.04

    # =====================================================
    # ESTADO
    # =====================================================

    param['trajectory'] = np.array(x)

    param['x'] = x


# =========================================================
# LOOP PRINCIPAL
# =========================================================

def loop():

    x = param['x']

    # obstáculos detectados
    ob = get_ob(x)

    # DWA
    u, predicted_trajectory = dw.dwa_control(

        x,
        config,
        config.goal,
        ob

    )

    # movimento real
    x = rap_motion(u)

    param['x'] = x

    param['trajectory'] = np.vstack(

        (param['trajectory'], x)

    )

    # =====================================================
    # VISUALIZAÇÃO
    # =====================================================

    if dw.show_animation:

        plt.cla()

        plt.plot(

            predicted_trajectory[:, 0],
            predicted_trajectory[:, 1],
            "-m"

        )

        plt.plot(
            x[0],
            x[1],
            "xr"
        )

        plt.plot(
            config.goal[0],
            config.goal[1],
            "xb"
        )

        plt.plot(
            ob[:, 0],
            ob[:, 1],
            "ok"
        )

        dw.plot_robot(
            x[0],
            x[1],
            x[2],
            config
        )

        dw.plot_arrow(
            x[0],
            x[1],
            x[2]
        )

        plt.axis("equal")

        plt.grid(True)

        plt.pause(0.001)

    # =====================================================
    # VERIFICA OBJETIVO
    # =====================================================

    dist_to_goal = math.hypot(

        x[0] - config.goal[0],
        x[1] - config.goal[1]

    )

    if dist_to_goal <= config.robot_radius:

        print("GOAL!")

        sim.setJointTargetVelocity(
            param['motorRight'],
            0
        )

        sim.setJointTargetVelocity(
            param['motorLeft'],
            0
        )

        return True

    return False


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    print("Iniciando simulação...")

    sim.setStepping(True)

    sim.startSimulation()

    init()

    try:

        while True:

            finished = loop()

            if finished:
                break

    except KeyboardInterrupt:

        print("Interrompido pelo usuário.")

    sim.stopSimulation()

    print("Simulação encerrada.")