import math

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import dynamic_window_approach as dw


client = RemoteAPIClient()
sim = client.require("sim")

param = {}
dwa = dw.DWAController()


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def get_required_object(path):
    try:
        return sim.getObject(path)
    except Exception as exc:
        raise RuntimeError(f"Objeto obrigatorio nao encontrado: {path}") from exc


def is_descendant_of(handle, parent):
    current = handle

    while current != -1:
        if current == parent:
            return True
        current = sim.getObjectParent(current)

    return False


def transform_point(matrix, point):
    return np.array(
        [
            matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
            matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
            matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
        ],
        dtype=float,
    )


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


def fill_rectangle_points(points, min_x, max_x, min_y, max_y, step=0.05):
    x = min_x

    while x <= max_x:
        y = min_y

        while y <= max_y:
            points.append([x, y])
            y += step

        x += step


def create_static_obstacles():
    points = []

    floor = get_required_object("/Floor")
    floor_pos = sim.getObjectPosition(floor, -1)
    floor_min_x = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x)
    floor_max_x = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x)
    floor_min_y = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y)
    floor_max_y = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y)

    add_rectangle_points(
        points,
        floor_pos[0] + floor_min_x,
        floor_pos[0] + floor_max_x,
        floor_pos[1] + floor_min_y,
        floor_pos[1] + floor_max_y,
    )

    for obj in sim.getObjectsInTree(sim.handle_scene):
        if sim.getObjectType(obj) != sim.object_shape_type:
            continue

        alias = sim.getObjectAlias(obj, 0)

        if alias in {"Floor", "box", "Goal", "Target"}:
            continue

        if obj == param["robot"] or is_descendant_of(obj, param["robot"]):
            continue

        pos = sim.getObjectPosition(obj, -1)
        min_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
        max_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)
        min_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
        max_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)

        width = max_x - min_x
        height = max_y - min_y

        if width > 3.0 and height > 3.0:
            continue

        fill_rectangle_points(points, min_x, max_x, min_y, max_y)

    unique_points = {}

    for x, y in points:
        unique_points[(round(x, 2), round(y, 2))] = [x, y]

    return np.array(list(unique_points.values()), dtype=float)


def get_robot_state(v=0.0, w=0.0):
    position = sim.getObjectPosition(param["robot"], -1)
    orientation = sim.getObjectOrientation(param["robot"], -1)

    return np.array(
        [
            position[0],
            position[1],
            normalize_angle(orientation[2]),
            v,
            w,
        ],
        dtype=float,
    )


def get_sensor_obstacles():
    obstacles = []

    for sensor in param["sensors"]:
        result, distance, point, obj, normal = sim.readProximitySensor(sensor)

        if result > 0:
            detected_point = np.array(point, dtype=float)

            if np.linalg.norm(detected_point) <= 0.0 and distance > 0.0:
                detected_point = np.array([distance, 0.0, 0.0], dtype=float)

            matrix = sim.getObjectMatrix(sensor, -1)
            obstacle_world = transform_point(matrix, detected_point)
            obstacles.append([obstacle_world[0], obstacle_world[1]])

    return np.array(obstacles, dtype=float)


def get_obstacles(x):
    local_obstacles = []
    static_obstacles = param.get("static_obstacles")

    if static_obstacles is not None and len(static_obstacles) > 0:
        distances = np.hypot(static_obstacles[:, 0] - x[0], static_obstacles[:, 1] - x[1])
        local_obstacles.extend(static_obstacles[distances <= 1.4].tolist())

    sensor_obstacles = get_sensor_obstacles()

    if len(sensor_obstacles) > 0:
        local_obstacles.extend(sensor_obstacles.tolist())

    return np.array(local_obstacles, dtype=float)


def would_collide(x):
    static_obstacles = param.get("static_obstacles")

    if static_obstacles is None or len(static_obstacles) == 0:
        return False

    distances = np.hypot(static_obstacles[:, 0] - x[0], static_obstacles[:, 1] - x[1])
    return float(np.min(distances)) <= dwa.robot_radius


def build_global_path():
    obstacles = param["static_obstacles"]
    planner = dw.AStarPlanner(
        obstacles[:, 0].tolist(),
        obstacles[:, 1].tolist(),
        resolution=0.1,
        rr=0.28,
    )
    rx, ry = planner.planning(
        float(param["x"][0]),
        float(param["x"][1]),
        float(param["goal_coords"][0]),
        float(param["goal_coords"][1]),
    )
    param["global_path"] = list(zip(rx, ry))
    param["path_index"] = 0
    print("Waypoints globais:", len(param["global_path"]))


def get_path_target(x):
    path = param.get("global_path", [param["goal_coords"]])

    while param["path_index"] < len(path) - 1:
        target = path[param["path_index"]]
        if math.hypot(target[0] - x[0], target[1] - x[1]) >= 0.45:
            break
        param["path_index"] += 1

    lookahead_index = min(param["path_index"] + 3, len(path) - 1)
    return path[lookahead_index]


def robot_motion(u):
    v = float(u[0])
    w = float(u[1])
    dt = dwa.dt
    wheel_radius = 0.0375
    wheel_base = 0.15

    x = param["x"].copy()
    x[2] = normalize_angle(x[2] + w * dt)
    x[0] += v * math.cos(x[2]) * dt
    x[1] += v * math.sin(x[2]) * dt
    x[3] = v
    x[4] = w

    if would_collide(x):
        x = param["x"].copy()
        x[2] = normalize_angle(x[2] + w * dt)
        x[3] = 0.0
        x[4] = w
        v = 0.0

    wr = (2.0 * v + w * wheel_base) / (2.0 * wheel_radius)
    wl = (2.0 * v - w * wheel_base) / (2.0 * wheel_radius)
    wr = max(min(wr, 20.0), -20.0)
    wl = max(min(wl, 20.0), -20.0)

    sim.setJointTargetVelocity(param["motorRight"], wr)
    sim.setJointTargetVelocity(param["motorLeft"], wl)
    sim.setObjectPosition(param["robot"], -1, [x[0], x[1], param["robot_z"]])
    sim.setObjectOrientation(param["robot"], -1, [param["robot_roll"], param["robot_pitch"], x[2]])

    try:
        sim.resetDynamicObject(param["robot"])
    except Exception:
        pass

    sim.step()

    return x


def init():
    param["motorRight"] = get_required_object("/MOTOR_DIREITO")
    param["motorLeft"] = get_required_object("/MOTOR_ESQUERDO")
    param["robot"] = sim.getObjectParent(param["motorRight"])
    param["goal"] = get_required_object("/Goal")

    param["sensors"] = [
        get_required_object("/SENSOR_MEIO"),
        get_required_object("/SENSOR_DIAG_DIREITO"),
        get_required_object("/SENSOR_DIAG_ESQUERDO"),
        get_required_object("/SENSOR_DIREITO"),
        get_required_object("/SENSOR_ESQUERDO"),
    ]

    robot_pos = sim.getObjectPosition(param["robot"], -1)
    robot_ori = sim.getObjectOrientation(param["robot"], -1)
    param["robot_z"] = robot_pos[2]
    param["robot_roll"] = robot_ori[0]
    param["robot_pitch"] = robot_ori[1]
    param["x"] = get_robot_state()

    goal_pos = sim.getObjectPosition(param["goal"], -1)
    param["goal_coords"] = [goal_pos[0], goal_pos[1]]
    param["static_obstacles"] = create_static_obstacles()
    param["step_count"] = 0
    build_global_path()

    print("Obstaculos estaticos:", len(param["static_obstacles"]))
    print("Estado inicial:", [round(float(v), 3) for v in param["x"]])
    print("Goal:", [round(float(v), 3) for v in param["goal_coords"]])


def loop():
    if param["goal"] is not None:
        goal_pos = sim.getObjectPosition(param["goal"], -1)
        param["goal_coords"] = [goal_pos[0], goal_pos[1]]

    x = param["x"]
    obstacles = get_obstacles(x)
    current_target = get_path_target(x)
    u, predicted_trajectory = dwa.plan(
        x[0:3],
        x[3],
        x[4],
        current_target,
        obstacles,
    )

    x = robot_motion(u)
    param["x"] = x
    param["step_count"] += 1

    dist_goal = math.hypot(
        x[0] - param["goal_coords"][0],
        x[1] - param["goal_coords"][1],
    )

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

    if dist_goal <= 0.20:
        print("GOAL ATINGIDO!")
        return True

    return False


if __name__ == "__main__":
    print("Iniciando simulacao...")

    sim.setStepping(True)
    sim.startSimulation()

    try:
        init()

        while True:
            if loop():
                break

    except KeyboardInterrupt:
        print("Parado pelo usuario.")

    finally:
        if "motorRight" in param and "motorLeft" in param:
            sim.setJointTargetVelocity(param["motorRight"], 0.0)
            sim.setJointTargetVelocity(param["motorLeft"], 0.0)

        sim.stopSimulation()
        print("Simulacao encerrada.")
