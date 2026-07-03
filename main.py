import math

# Biblioteca para cálculos numéricos e vetoriais
import numpy as np

# Cliente da API remota do CoppeliaSim
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Arquivo onde está implementado o algoritmo DWA e o A* melhorado
import dynamic_window_approach as dw

# Módulo do Mapa de Grade de Ocupação (sensor de visão)
import mapa_ocupacao as mo


# -----------------------------------------------------------
# Override manual da direção "para frente".
# A frente é detectada automaticamente pelo sensor dianteiro
# (SENSOR_MEIO) no init(). Se, ao testar, o robô ainda andar
# "de costas", troque este valor para -1 (inverte 180°).
# -----------------------------------------------------------
SINAL_FRENTE = +1


# Cria conexão com o CoppeliaSim
client = RemoteAPIClient()

# Obtém acesso ao módulo principal da simulação
sim = client.require("sim")


# Dicionário usado para armazenar parâmetros globais do robô
param = {}

# Instancia o controlador Dynamic Window Approach
dwa = dw.DWAController()


# -----------------------------------------------------------
# Normaliza um ângulo para o intervalo [-pi, pi]
# Evita problemas de rotação acumulada
# -----------------------------------------------------------
def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# -----------------------------------------------------------
# Obtém um objeto da cena pelo caminho.
# Caso o objeto não exista, gera erro.
# -----------------------------------------------------------
def get_required_object(path):
    try:
        return sim.getObject(path)

    except Exception as exc:
        raise RuntimeError(f"Objeto obrigatorio nao encontrado: {path}") from exc


# -----------------------------------------------------------
# Verifica se um objeto é filho/descendente de outro.
# Usado para ignorar partes do próprio robô na detecção.
# -----------------------------------------------------------
def is_descendant_of(handle, parent):

    current = handle

    while current != -1:

        # Se encontrou o pai
        if current == parent:
            return True

        # Sobe um nível na hierarquia
        current = sim.getObjectParent(current)

    return False


# -----------------------------------------------------------
# Converte um ponto do sistema local para o sistema global.
# Usa matriz de transformação homogênea.
# -----------------------------------------------------------
def transform_point(matrix, point):

    return np.array(
        [
            matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],

            matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],

            matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
        ],
        dtype=float,
    )


# -----------------------------------------------------------
# Adiciona apenas as bordas de um retângulo.
# Isso é usado principalmente para representar limites.
# -----------------------------------------------------------
def add_rectangle_points(points, min_x, max_x, min_y, max_y, step=0.06):

    x = min_x

    while x <= max_x:

        points.append([x, min_y])
        points.append([x, max_y])

        x += step

    y = min_y

    while y <= max_y:

        points.append([min_x, y])
        points.append([max_x, y])

        y += step


# -----------------------------------------------------------
# Preenche completamente um retângulo com pontos.
# Esses pontos representam obstáculos ocupando espaço.
# -----------------------------------------------------------
def fill_rectangle_points(points, min_x, max_x, min_y, max_y, step=0.05):

    x = min_x

    while x <= max_x:

        y = min_y

        while y <= max_y:

            points.append([x, y])

            y += step

        x += step


# -----------------------------------------------------------
# Cria o mapa de obstáculos estáticos da cena.
# Analisa todos os objetos do ambiente.
# -----------------------------------------------------------
def create_static_obstacles():

    points = []

    # Obtém o chão da cena
    floor = get_required_object("/Floor")

    floor_pos = sim.getObjectPosition(floor, -1)

    floor_min_x = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x)
    floor_max_x = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x)

    floor_min_y = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y)
    floor_max_y = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y)

    # Adiciona bordas do chão
    add_rectangle_points(
        points,
        floor_pos[0] + floor_min_x,
        floor_pos[0] + floor_max_x,
        floor_pos[1] + floor_min_y,
        floor_pos[1] + floor_max_y,
    )

    # Percorre todos os objetos da cena
    for obj in sim.getObjectsInTree(sim.handle_scene):

        # Considera apenas objetos do tipo shape
        if sim.getObjectType(obj) != sim.object_shape_type:
            continue

        alias = sim.getObjectAlias(obj, 0)

        # Ignora objetos específicos
        if alias in {"Floor", "box", "Goal", "Target"}:
            continue

        # Ignora partes do robô
        if obj == param["robot"] or is_descendant_of(obj, param["robot"]):
            continue

        # Obtém posição e limites do objeto
        pos = sim.getObjectPosition(obj, -1)

        min_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
        max_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)

        min_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
        max_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)

        width = max_x - min_x
        height = max_y - min_y

        # Ignora objetos gigantes
        if width > 3.0 and height > 3.0:
            continue

        # Preenche o obstáculo com pontos
        fill_rectangle_points(points, min_x, max_x, min_y, max_y)

    # Remove pontos repetidos
    unique_points = {}

    for x, y in points:
        unique_points[(round(x, 2), round(y, 2))] = [x, y]

    return np.array(list(unique_points.values()), dtype=float)


# -----------------------------------------------------------
# Obtém o estado atual do robô.
# x, y, theta, velocidade linear e angular.
# -----------------------------------------------------------
def get_robot_state(v=0.0, w=0.0):

    position = sim.getObjectPosition(param["robot"], -1)
    orientation = sim.getObjectOrientation(param["robot"], -1)

    # Converte o yaw armazenado no CoppeliaSim para o rumo FÍSICO da frente
    # do robô (onde os sensores apontam), usando o offset medido no init().
    # Isso corrige o problema de o robô "andar de costas".
    heading_offset = param.get("heading_offset", 0.0)
    flip = 0.0 if SINAL_FRENTE >= 0 else math.pi
    theta_front = normalize_angle(orientation[2] + heading_offset + flip)

    return np.array(
        [
            position[0],
            position[1],
            theta_front,
            v,
            w,
        ],
        dtype=float,
    )


# -----------------------------------------------------------
# Mede o offset entre o yaw do modelo (Euler) e a direção
# física da frente do robô, definida pelo sensor dianteiro.
# Sensores de proximidade detectam ao longo do +Z local.
# -----------------------------------------------------------
def medir_heading_offset():

    sensor_frontal = param["sensors"][0]  # SENSOR_MEIO

    matrix = sim.getObjectMatrix(sensor_frontal, -1)

    # Eixo +Z do sensor no mundo = direção de detecção (a frente).
    front_x = matrix[2]
    front_y = matrix[6]
    front_yaw = math.atan2(front_y, front_x)

    yaw_modelo = sim.getObjectOrientation(param["robot"], -1)[2]

    param["heading_offset"] = normalize_angle(front_yaw - yaw_modelo)

    print(
        "Heading offset (frente - yaw):",
        round(math.degrees(param["heading_offset"]), 1),
        "graus",
    )


# -----------------------------------------------------------
# Lê os sensores de proximidade do robô.
# Retorna obstáculos detectados em coordenadas globais.
# -----------------------------------------------------------
def get_sensor_obstacles():

    obstacles = []

    for sensor in param["sensors"]:

        result, distance, point, obj, normal = sim.readProximitySensor(sensor)

        # Se detectou algo
        if result > 0:

            detected_point = np.array(point, dtype=float)

            # Corrige casos onde o ponto vem zerado
            if np.linalg.norm(detected_point) <= 0.0 and distance > 0.0:
                detected_point = np.array([distance, 0.0, 0.0], dtype=float)

            # Converte coordenadas locais para globais
            matrix = sim.getObjectMatrix(sensor, -1)

            obstacle_world = transform_point(matrix, detected_point)

            obstacles.append([obstacle_world[0], obstacle_world[1]])

    return np.array(obstacles, dtype=float)


# -----------------------------------------------------------
# Junta obstáculos estáticos e detectados por sensores.
# Apenas obstáculos próximos do robô são considerados.
# -----------------------------------------------------------
def get_obstacles(x):

    local_obstacles = []

    static_obstacles = param.get("static_obstacles")

    # Obstáculos estáticos próximos
    if static_obstacles is not None and len(static_obstacles) > 0:

        distances = np.hypot(
            static_obstacles[:, 0] - x[0],
            static_obstacles[:, 1] - x[1]
        )

        local_obstacles.extend(
            static_obstacles[distances <= 1.4].tolist()
        )

    # Obstáculos vindos dos sensores
    sensor_obstacles = get_sensor_obstacles()

    if len(sensor_obstacles) > 0:
        local_obstacles.extend(sensor_obstacles.tolist())

    return np.array(local_obstacles, dtype=float)


# -----------------------------------------------------------
# Verifica se a posição prevista colidiria com algum obstáculo.
# -----------------------------------------------------------
def would_collide(x):

    static_obstacles = param.get("static_obstacles")

    if static_obstacles is None or len(static_obstacles) == 0:
        return False

    distances = np.hypot(
        static_obstacles[:, 0] - x[0],
        static_obstacles[:, 1] - x[1]
    )

    return float(np.min(distances)) <= dwa.collision_radius


# -----------------------------------------------------------
# Cria caminho global usando algoritmo A*.
# -----------------------------------------------------------
def build_global_path():

    obstacles = param["static_obstacles"]

    sx, sy = float(param["x"][0]), float(param["x"][1])
    gx, gy = float(param["goal_coords"][0]), float(param["goal_coords"][1])

    # Margem adaptativa: tenta primeiro uma inflação conservadora e vai
    # reduzindo caso não encontre rota (corredores estreitos / objetivo
    # colado na parede). O piso 0.20 casa com o raio de colisão do DWA.
    rx = ry = kx = ky = None
    for rr in (0.24, 0.20, 0.16):
        planner = dw.AStarPlanner(
            obstacles[:, 0].tolist(),
            obstacles[:, 1].tolist(),
            resolution=0.1,
            rr=rr,
        )
        # O A* melhorado devolve:
        #   rx, ry -> caminho denso suavizado por Bézier (rastreamento)
        #   kx, ky -> pontos-chave (metas intermediárias do DWA)
        rx, ry, kx, ky = planner.planning(sx, sy, gx, gy)

        if not planner.last_plan_failed:
            print(f"A* encontrou rota com margem rr={rr:.2f} m")
            break

        print(f"A* sem rota com rr={rr:.2f} m; reduzindo margem...")
    else:
        print("A* não encontrou rota mesmo com margem mínima; seguindo em linha reta.")
        # Diagnóstico: onde está o gargalo?
        if len(obstacles) > 0:
            d_start = float(np.min(np.hypot(obstacles[:, 0] - sx, obstacles[:, 1] - sy)))
            d_goal = float(np.min(np.hypot(obstacles[:, 0] - gx, obstacles[:, 1] - gy)))
            print(f"   diag: obstáculo mais próximo do INÍCIO = {d_start*100:.0f} cm, "
                  f"do GOAL = {d_goal*100:.0f} cm (raio de colisão = {dwa.collision_radius*100:.0f} cm)")
            if d_goal < dwa.collision_radius:
                print("   -> O GOAL está praticamente encostado/dentro de um obstáculo. "
                      "Afaste o Goal da parede ~20 cm ou reduza collision_radius.")
            if d_start < dwa.collision_radius:
                print("   -> O INÍCIO do robô está colado a um obstáculo (talvez o próprio corpo "
                      "na grade). Verifique a exclusão do robô / posição inicial.")

    # Armazena o caminho suavizado (usado no lookahead) e os pontos-chave.
    param["global_path"] = list(zip(rx, ry))
    param["key_points"] = list(zip(kx, ky))

    param["path_index"] = 0

    print("Waypoints (caminho suavizado):", len(param["global_path"]))
    print("Pontos-chave (A* melhorado):", len(param["key_points"]))


# -----------------------------------------------------------
# Define qual waypoint o robô deve seguir no momento.
# -----------------------------------------------------------
def get_path_target(x):

    path = param.get("global_path", [param["goal_coords"]])

    # Avança no caminho conforme o robô chega perto
    while param["path_index"] < len(path) - 1:

        target = path[param["path_index"]]

        if math.hypot(target[0] - x[0], target[1] - x[1]) >= 0.45:
            break

        param["path_index"] += 1

    # Lookahead para suavizar movimento
    lookahead_index = min(param["path_index"] + 3, len(path) - 1)

    return path[lookahead_index]


# -----------------------------------------------------------
# Aplica movimento no robô.
# Converte velocidades linear/angular em velocidades das rodas.
# -----------------------------------------------------------
def robot_motion(u):

    v = float(u[0])  # velocidade linear
    w = float(u[1])  # velocidade angular

    dt = dwa.dt

    wheel_radius = 0.0375
    wheel_base = 0.15

    # Copia estado atual
    x = param["x"].copy()

    # Atualiza orientação
    x[2] = normalize_angle(x[2] + w * dt)

    # Atualiza posição
    x[0] += v * math.cos(x[2]) * dt
    x[1] += v * math.sin(x[2]) * dt

    x[3] = v
    x[4] = w

    # Se houver colisão, impede movimento linear
    if would_collide(x):

        x = param["x"].copy()

        x[2] = normalize_angle(x[2] + w * dt)

        x[3] = 0.0
        x[4] = w

        v = 0.0

    # Cinemática diferencial
    wr = (2.0 * v + w * wheel_base) / (2.0 * wheel_radius)
    wl = (2.0 * v - w * wheel_base) / (2.0 * wheel_radius)

    # Limita velocidade das rodas
    wr = max(min(wr, 20.0), -20.0)
    wl = max(min(wl, 20.0), -20.0)

    # Envia velocidades aos motores
    sim.setJointTargetVelocity(param["motorRight"], wr)
    sim.setJointTargetVelocity(param["motorLeft"], wl)

    # Atualiza posição do robô na simulação
    sim.setObjectPosition(
        param["robot"],
        -1,
        [x[0], x[1], param["robot_z"]]
    )

    # Converte o rumo da frente (x[2]) de volta para o yaw do modelo antes
    # de escrever a orientação no simulador.
    heading_offset = param.get("heading_offset", 0.0)
    flip = 0.0 if SINAL_FRENTE >= 0 else math.pi
    yaw_modelo = normalize_angle(x[2] - heading_offset - flip)

    sim.setObjectOrientation(
        param["robot"],
        -1,
        [param["robot_roll"], param["robot_pitch"], yaw_modelo]
    )

    # Reinicia dinâmica do objeto
    try:
        sim.resetDynamicObject(param["robot"])

    except Exception:
        pass

    # Avança um passo da simulação
    sim.step()

    return x


# -----------------------------------------------------------
# Obtém os obstáculos estáticos a partir do Mapa de Grade de
# Ocupação (sensor de visão). Se falhar (ex.: sensor não pôde
# ser criado), recorre ao antigo scan por bounding box.
# -----------------------------------------------------------
def obter_obstaculos_estaticos():

    # Discos a excluir da grade: início do robô e objetivo, para não
    # bloquear a origem/destino do A*.
    excluir = [
        (float(param["x"][0]), float(param["x"][1]), dwa.robot_radius + 0.20),
        (float(param["goal_coords"][0]), float(param["goal_coords"][1]), dwa.robot_radius + 0.10),
    ]

    try:
        floor = get_required_object("/Floor")

        pontos, grade, info = mo.construir_obstaculos_por_visao(
            sim, floor, excluir=excluir
        )

        param["occ_grade"] = grade
        param["occ_info"] = info

        ocupadas = int(np.count_nonzero(grade))

        print(
            "Grade de ocupação:",
            f"{grade.shape[1]}x{grade.shape[0]} células,",
            f"{ocupadas} ocupadas,",
            f"célula={info['passo']*100:.1f} cm,",
            f"{len(pontos)} pontos de obstáculo",
        )

        if len(pontos) >= 4:
            return pontos

        print("Grade de ocupação vazia; usando scan por bounding box.")

    except Exception as exc:
        print("Falha na grade de ocupação (", exc, "); usando bounding box.")

    return create_static_obstacles()


# -----------------------------------------------------------
# Inicializa parâmetros, sensores, robô e obstáculos.
# -----------------------------------------------------------
def init():

    # Motores
    param["motorRight"] = get_required_object("/MOTOR_DIREITO")
    param["motorLeft"] = get_required_object("/MOTOR_ESQUERDO")

    # Robô
    param["robot"] = sim.getObjectParent(param["motorRight"])

    # Goal
    param["goal"] = get_required_object("/Goal")

    # Sensores
    param["sensors"] = [
        get_required_object("/SENSOR_MEIO"),
        get_required_object("/SENSOR_DIAG_DIREITO"),
        get_required_object("/SENSOR_DIAG_ESQUERDO"),
        get_required_object("/SENSOR_DIREITO"),
        get_required_object("/SENSOR_ESQUERDO"),
    ]

    # Estado inicial do robô
    robot_pos = sim.getObjectPosition(param["robot"], -1)

    robot_ori = sim.getObjectOrientation(param["robot"], -1)

    param["robot_z"] = robot_pos[2]

    param["robot_roll"] = robot_ori[0]
    param["robot_pitch"] = robot_ori[1]

    # Mede o offset de rumo ANTES de ler o estado (corrige "andar de costas").
    medir_heading_offset()

    param["x"] = get_robot_state()

    # Coordenadas do objetivo
    goal_pos = sim.getObjectPosition(param["goal"], -1)

    param["goal_coords"] = [goal_pos[0], goal_pos[1]]

    # Gera obstáculos do ambiente a partir do MAPA DE GRADE DE OCUPAÇÃO
    # (sensor de visão). Recorre ao scan por bounding box apenas como fallback.
    param["static_obstacles"] = obter_obstaculos_estaticos()

    param["step_count"] = 0

    # Cria rota global
    build_global_path()

    print("Obstaculos estaticos:", len(param["static_obstacles"]))

    print(
        "Estado inicial:",
        [round(float(v), 3) for v in param["x"]]
    )

    print(
        "Goal:",
        [round(float(v), 3) for v in param["goal_coords"]]
    )


# -----------------------------------------------------------
# Replaneja a rota global a partir da posição ATUAL do robô.
# Chamado quando o robô fica preso, pois o A* inicial pode ter
# falhado (rota inexistente na época) ou ficado desatualizado.
# -----------------------------------------------------------
def replanejar():
    build_global_path()
    print("Replanejado a partir de", [round(float(param["x"][0]), 2), round(float(param["x"][1]), 2)])


# -----------------------------------------------------------
# Manobra anti-travamento (escape de mínimo local):
# recua para se afastar da parede à frente e gira em direção
# ao alvo, até conseguir prosseguir.
# -----------------------------------------------------------
def comando_recuperacao(x, target):
    ang = math.atan2(target[1] - x[1], target[0] - x[0])
    turn = normalize_angle(ang - x[2])
    w = max(min(1.2 * turn, dwa.max_yaw_rate), -dwa.max_yaw_rate)

    fase = param.get("_recovery", 0)

    if fase > 22:
        # Fase 1: recua um pouco para descolar da parede.
        return [-0.08, 0.3 * w]

    if abs(turn) > 0.30:
        # Fase 2: gira no próprio eixo em direção ao alvo.
        return [0.0, 0.9 if turn >= 0.0 else -0.9]

    # Fase 3: alinhado — empurra para frente.
    return [0.12, 0.5 * w]


# -----------------------------------------------------------
# Loop principal da navegação.
# -----------------------------------------------------------
def loop():

    # Atualiza posição do goal caso ele se mova
    if param["goal"] is not None:

        goal_pos = sim.getObjectPosition(param["goal"], -1)

        param["goal_coords"] = [goal_pos[0], goal_pos[1]]

    x = param["x"]

    # Obtém obstáculos próximos
    obstacles = get_obstacles(x)

    # Obtém waypoint atual
    current_target = get_path_target(x)

    # --- Detecção de travamento (mínimo local) ---
    ref = param.get("_stuck_ref")
    if ref is None or math.hypot(x[0] - ref[0], x[1] - ref[1]) > 0.06:
        param["_stuck_ref"] = (float(x[0]), float(x[1]))
        param["_stuck_steps"] = 0
    else:
        param["_stuck_steps"] = param.get("_stuck_steps", 0) + 1

    if param.get("_recovery", 0) > 0:
        # Em recuperação: usa a manobra de escape, não o DWA.
        u = comando_recuperacao(x, current_target)
        param["_recovery"] -= 1
    else:
        # DWA calcula o melhor comando.
        u, predicted_trajectory = dwa.plan(
            x[0:3],
            x[3],
            x[4],
            current_target,
            obstacles,
        )

        # Preso há muitos passos sem progredir: replaneja e recupera.
        if param.get("_stuck_steps", 0) >= 30:
            print("Robô preso -> replanejando e iniciando recuperação")
            replanejar()
            param["_recovery"] = 35
            param["_stuck_steps"] = 0

    # Move robô
    x = robot_motion(u)

    param["x"] = x

    param["step_count"] += 1

    # Distância até o objetivo
    dist_goal = math.hypot(
        x[0] - param["goal_coords"][0],
        x[1] - param["goal_coords"][1],
    )

    # Debug a cada 10 passos
    if param["step_count"] % 10 == 0:

        print(
            "step",
            param["step_count"],
            "dist",
            round(dist_goal, 2),
            "pos",
            [round(float(x[0]), 2), round(float(x[1]), 2)],
            "u",
            [round(float(u[0]), 2), round(float(u[1]), 2)],
            "wp",
            param.get("path_index", 0),
            "obs",
            len(obstacles),
        )

    # Verifica se chegou ao objetivo
    if dist_goal <= 0.20:

        print("GOAL ATINGIDO!")

        return True

    return False


# -----------------------------------------------------------
# Programa principal
# -----------------------------------------------------------
if __name__ == "__main__":

    print("Iniciando simulacao...")

    # Simulação em modo step-by-step
    sim.setStepping(True)

    sim.startSimulation()

    try:

        # Inicialização
        init()

        # Loop infinito
        while True:

            if loop():
                break

    # Interrompe com CTRL+C
    except KeyboardInterrupt:

        print("Parado pelo usuario.")

    finally:

        # Para motores
        if "motorRight" in param and "motorLeft" in param:

            sim.setJointTargetVelocity(param["motorRight"], 0.0)
            sim.setJointTargetVelocity(param["motorLeft"], 0.0)

        # Encerra simulação
        sim.stopSimulation()

        print("Simulacao encerrada.")