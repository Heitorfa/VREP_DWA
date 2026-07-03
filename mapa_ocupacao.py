"""
Mapa de Grade de Ocupacao (Occupancy Grid) por sensor de visao.

Requisito do professor: construir a grade de ocupacao da cena no CoppeliaSim
de forma simplificada, obtendo uma IMAGEM da cena com um sensor de visao
posicionado acima da superficie e classificando cada celula como LIVRE ou
OCUPADA.

Referencia base:
    H. Moravec; A. E. Elfes (1985). "High resolution maps from wide angle
    sonar". Proc. 1985 IEEE Int. Conf. on Robotics and Automation, pp. 116-121.

Estrategia (equivalente 2D, top-down, do modelo de ocupacao de Elfes):
  * Cria-se por CODIGO (API) um sensor de visao ORTOGRAFICO apontado para
    baixo, centralizado sobre o /Floor, cobrindo toda a extensao da cena.
  * Captura-se o mapa de PROFUNDIDADE. Onde a superficie medida esta mais
    perto do sensor do que o chao (ou seja, existe algo "alto" naquela
    celula), a celula e marcada como OCUPADA.
  * Ha tambem um metodo alternativo por INTENSIDADE (imagem em tons de cinza)
    caso o metodo de profundidade nao seja desejado.
  * As celulas ocupadas sao convertidas em pontos (x, y) no referencial do
    mundo, que alimentam o planejador A* (substituindo o antigo scan por
    bounding box).

Este modulo pode ser executado isoladamente para depuracao:
    python mapa_ocupacao.py
"""

import math

import numpy as np


# ==========================================================================
#  CONFIGURACAO
# ==========================================================================
class ConfigGrade:
    # Nome do sensor de visao criado na cena (reutilizado se ja existir).
    nome_sensor = "/OccupancyCam"

    # Resolucao da imagem/grade (NxN celulas).
    resolucao = 256

    # Altura (m) do sensor acima do plano do chao.
    altura = 3.0

    # Margem (m) adicionada a extensao do chao para garantir cobertura total.
    margem_cobertura = 0.2

    # Altura minima (m) para considerar um objeto como obstaculo. Superficies
    # mais baixas que isso (ex.: o proprio chao) sao consideradas livres.
    altura_min_obstaculo = 0.05

    # Espelhamentos da imagem -> mundo. Ajuste caso a grade saia invertida
    # em relacao a cena (verificavel rodando este modulo isoladamente).
    flip_x = False
    flip_y = True


# ==========================================================================
#  CRIACAO / OBTENCAO DO SENSOR DE VISAO (por codigo)
# ==========================================================================
def criar_ou_obter_sensor(sim, cfg, centro_xy, tamanho_cena):
    """Cria (ou reutiliza) o sensor de visao ortografico apontado para baixo.

    Parametros
    ----------
    centro_xy      : (cx, cy) centro do chao no mundo.
    tamanho_cena   : lado (m) da regiao quadrada a cobrir.
    Retorna o handle do sensor.
    """
    # Reutiliza se ja existir na cena.
    try:
        sensor = sim.getObject(cfg.nome_sensor)
        _configurar_sensor(sim, sensor, cfg, centro_xy, tamanho_cena)
        return sensor
    except Exception:
        pass

    # Cria um novo sensor de visao.
    #   intParams:  [resX, resY, 0, 0]
    #   floatParams:[near, far, view/ortho, sx, sy, sz, r, g, b, 0, 0]
    int_params = [cfg.resolucao, cfg.resolucao, 0, 0]
    float_params = [
        0.01,                       # near clipping
        cfg.altura + 1.0,           # far clipping
        tamanho_cena,               # tamanho ortografico
        0.05, 0.05, 0.02,           # tamanho visual do sensor
        0.0, 0.0, 0.0, 0.0, 0.0,    # cor de pixel nulo / reservados
    ]
    # options=1 -> sensor explicitamente manipulado (sim.handleVisionSensor).
    sensor = sim.createVisionSensor(1, int_params, float_params)
    sim.setObjectAlias(sensor, cfg.nome_sensor.strip("/"))
    _configurar_sensor(sim, sensor, cfg, centro_xy, tamanho_cena)
    return sensor


def _configurar_sensor(sim, sensor, cfg, centro_xy, tamanho_cena):
    """Garante modo ortografico, resolucao, clipping, posicao e orientacao."""
    cx, cy = centro_xy

    # Forca modo ORTOGRAFICO e parametros (mais confiavel que os bits do create).
    try:
        sim.setObjectInt32Param(sensor, sim.visionintparam_perspective_operation, 0)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_x, cfg.resolucao)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_y, cfg.resolucao)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_ortho_size, tamanho_cena)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_near_clipping, 0.01)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_far_clipping, cfg.altura + 1.0)
    except Exception as exc:
        print("[mapa_ocupacao] Aviso ao configurar parametros do sensor:", exc)

    # Posiciona acima do centro da cena, olhando para baixo (-Z do mundo).
    sim.setObjectPosition(sensor, -1, [cx, cy, cfg.altura])
    sim.setObjectOrientation(sensor, -1, [math.pi, 0.0, 0.0])


# ==========================================================================
#  EXTENSAO DO CHAO
# ==========================================================================
def extensao_do_chao(sim, floor_handle):
    """Retorna (centro_xy, tamanho) da regiao quadrada que cobre o chao."""
    pos = sim.getObjectPosition(floor_handle, -1)
    min_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_x)
    max_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_x)
    min_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_y)
    max_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_y)

    largura = (max_x - min_x)
    altura = (max_y - min_y)
    centro = (pos[0], pos[1])
    tamanho = max(largura, altura)
    return centro, tamanho


# ==========================================================================
#  CAPTURA DA GRADE DE OCUPACAO
# ==========================================================================
def capturar_grade(sim, sensor, cfg):
    """Captura o mapa de profundidade e devolve a grade booleana (ocupada?).

    Retorna (grade, resolucao) onde grade[row][col] == True significa OCUPADA.
    A profundidade normalizada [0,1] e convertida em metros entre os planos
    near e far.
    """
    # Em modo stepping, e necessario processar o sensor antes de ler.
    sim.handleVisionSensor(sensor)

    near = cfg.altura and 0.01
    far = cfg.altura + 1.0

    buffer, resolucao = sim.getVisionSensorDepth(sensor, 1)
    depth = np.array(sim.unpackFloatTable(buffer), dtype=float)
    nx, ny = int(resolucao[0]), int(resolucao[1])
    depth = depth.reshape((ny, nx))

    # Profundidade normalizada -> metros (distancia sensor->superficie).
    dist_m = near + depth * (far - near)

    # Ocupada onde a superficie esta acima do chao mais que o limiar:
    #   dist ao chao ~ altura; objeto de altura h -> dist ~ altura - h.
    limiar = cfg.altura - cfg.altura_min_obstaculo
    grade = dist_m < limiar
    return grade, (nx, ny)


def capturar_grade_por_intensidade(sim, sensor, cfg, limiar=0.5, obstaculo_escuro=True):
    """Alternativa: gera a grade a partir da imagem em tons de cinza.

    Assume contraste entre chao e obstaculos. Se obstaculo_escuro=True, pixels
    mais escuros que o limiar sao considerados ocupados.
    """
    sim.handleVisionSensor(sensor)
    img, resolucao = sim.getVisionSensorImg(sensor)
    nx, ny = int(resolucao[0]), int(resolucao[1])
    arr = np.frombuffer(img, dtype=np.uint8).reshape((ny, nx, 3)).astype(float) / 255.0
    cinza = arr.mean(axis=2)
    if obstaculo_escuro:
        return cinza < limiar, (nx, ny)
    return cinza > limiar, (nx, ny)


# ==========================================================================
#  CONVERSAO GRADE -> PONTOS DE OBSTACULO NO MUNDO
# ==========================================================================
def grade_para_pontos(grade, cfg, centro_xy, tamanho_cena, excluir=None):
    """Converte celulas ocupadas em pontos (x, y) no referencial do mundo.

    excluir: lista de (x, y, raio) a remover (ex.: posicao do robo e do Goal),
             para nao bloquear o inicio/objetivo do A*.
    """
    ny, nx = grade.shape
    cx, cy = centro_xy
    passo = tamanho_cena / nx  # tamanho de cada celula (m)

    pontos = []
    ocupadas = np.argwhere(grade)
    for row, col in ocupadas:
        c = (nx - 1 - col) if cfg.flip_x else col
        r = (ny - 1 - row) if cfg.flip_y else row
        wx = cx + (c + 0.5 - nx / 2.0) * passo
        wy = cy + (r + 0.5 - ny / 2.0) * passo
        pontos.append([wx, wy])

    pontos = np.array(pontos, dtype=float) if pontos else np.empty((0, 2), dtype=float)

    if excluir and len(pontos) > 0:
        for ex, ey, er in excluir:
            d = np.hypot(pontos[:, 0] - ex, pontos[:, 1] - ey)
            pontos = pontos[d > er]

    return pontos


# ==========================================================================
#  API DE ALTO NIVEL (usada pelo main.py)
# ==========================================================================
def construir_obstaculos_por_visao(sim, floor_handle, cfg=None, excluir=None):
    """Fluxo completo: cria sensor, captura grade e devolve pontos de obstaculo.

    Substitui a antiga funcao create_static_obstacles() baseada em bounding box.
    Retorna (pontos_obstaculo, grade, info) onde info traz centro/tamanho/passo.
    """
    if cfg is None:
        cfg = ConfigGrade()

    centro, tamanho = extensao_do_chao(sim, floor_handle)
    tamanho += cfg.margem_cobertura
    sensor = criar_ou_obter_sensor(sim, cfg, centro, tamanho)

    try:
        grade, _ = capturar_grade(sim, sensor, cfg)
    except Exception as exc:
        print("[mapa_ocupacao] Profundidade falhou (", exc, "); usando intensidade.")
        grade, _ = capturar_grade_por_intensidade(sim, sensor, cfg)

    pontos = grade_para_pontos(grade, cfg, centro, tamanho, excluir=excluir)
    info = {
        "centro": centro,
        "tamanho": tamanho,
        "passo": tamanho / grade.shape[1],
        "sensor": sensor,
    }
    return pontos, grade, info


# ==========================================================================
#  FALLBACK: OBSTACULOS POR BOUNDING BOX (varredura da arvore da cena)
#  Usado apenas se a grade de ocupacao por visao falhar.
# ==========================================================================
def _descende_de(sim, handle, ancestral):
    atual = handle
    while atual != -1:
        if atual == ancestral:
            return True
        atual = sim.getObjectParent(atual)
    return False


def _preenche_retangulo(pontos, min_x, max_x, min_y, max_y, passo=0.05):
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            pontos.append([x, y])
            y += passo
        x += passo


def _bordas_retangulo(pontos, min_x, max_x, min_y, max_y, passo=0.06):
    x = min_x
    while x <= max_x:
        pontos.append([x, min_y])
        pontos.append([x, max_y])
        x += passo
    y = min_y
    while y <= max_y:
        pontos.append([min_x, y])
        pontos.append([max_x, y])
        y += passo


def construir_obstaculos_por_bbox(sim, robot_handle, ignorar=("Floor", "box", "Goal", "Target")):
    """Varre a arvore da cena e converte bounding boxes em pontos 2D."""
    pontos = []

    floor = sim.getObject("/Floor")
    fp = sim.getObjectPosition(floor, -1)
    _bordas_retangulo(
        pontos,
        fp[0] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x),
        fp[0] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x),
        fp[1] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y),
        fp[1] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y),
    )

    for obj in sim.getObjectsInTree(sim.handle_scene):
        if sim.getObjectType(obj) != sim.object_shape_type:
            continue
        if sim.getObjectAlias(obj, 0) in ignorar:
            continue
        if obj == robot_handle or _descende_de(sim, obj, robot_handle):
            continue

        pos = sim.getObjectPosition(obj, -1)
        min_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
        max_x = pos[0] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)
        min_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
        max_y = pos[1] + sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)

        if (max_x - min_x) > 3.0 and (max_y - min_y) > 3.0:
            continue  # ignora objetos gigantes (ceu, camera rig etc.)

        _preenche_retangulo(pontos, min_x, max_x, min_y, max_y)

    unicos = {(round(x, 2), round(y, 2)): [x, y] for x, y in pontos}
    return np.array(list(unicos.values()), dtype=float)


# ==========================================================================
#  EXECUCAO ISOLADA (DEPURACAO)
# ==========================================================================
def _ascii_preview(grade, cols=60):
    """Imprime uma previa em ASCII da grade (subamostrada)."""
    ny, nx = grade.shape
    passo = max(1, nx // cols)
    linhas = []
    for row in range(0, ny, passo):
        linha = "".join(
            "#" if grade[row, col] else "." for col in range(0, nx, passo)
        )
        linhas.append(linha)
    print("\n".join(linhas))


if __name__ == "__main__":
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print("Conectando ao CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.require("sim")

    cfg = ConfigGrade()

    sim.setStepping(True)
    sim.startSimulation()
    try:
        floor = sim.getObject("/Floor")
        pontos, grade, info = construir_obstaculos_por_visao(sim, floor, cfg)

        ocupadas = int(np.count_nonzero(grade))
        total = grade.size
        print(f"Resolucao da grade : {grade.shape[1]} x {grade.shape[0]}")
        print(f"Centro da cena     : {info['centro']}")
        print(f"Tamanho coberto    : {info['tamanho']:.2f} m")
        print(f"Tamanho da celula  : {info['passo']*100:.1f} cm")
        print(f"Celulas ocupadas   : {ocupadas}/{total} ({100.0*ocupadas/total:.1f}%)")
        print(f"Pontos de obstaculo: {len(pontos)}")
        print("\nPrevia (ASCII, subamostrada):")
        _ascii_preview(grade)

        # Salva uma imagem PNG da grade, se matplotlib estiver disponivel.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.imshow(grade, cmap="Greys", origin="lower")
            plt.title("Grade de Ocupacao (sensor de visao)")
            plt.tight_layout()
            plt.savefig("grade_ocupacao.png", dpi=120)
            print("\nImagem salva em grade_ocupacao.png")
        except Exception as exc:
            print("(matplotlib indisponivel para salvar PNG:", exc, ")")

    finally:
        sim.stopSimulation()
        print("Simulacao encerrada.")
