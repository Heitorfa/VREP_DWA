"""
Navegacao autonoma no CoppeliaSim: A* MELHORADO (global) + DWA (local).

Pipeline de cada simulacao:
  1. Conecta ao CoppeliaSim (ZeroMQ Remote API) e inicia em modo stepping.
  2. Constroi o MAPA DE GRADE DE OCUPACAO com um sensor de visao criado por
     codigo acima da cena (mapa_ocupacao.py). Fallback: bounding boxes.
  3. Planeja a rota global com o A* melhorado (Guo et al., 2024) com margem
     adaptativa e penalidade de proximidade (dynamic_window_approach.py).
  4. Exibe o MAPA DE OCUPACAO + ROTA em uma janela ao vivo (visualizacao.py)
     e salva mapa_rota.png ao final.
  5. Navega com o DWA usando os pontos do caminho como metas locais, com
     leitura reativa dos sensores de proximidade, deteccao de travamento,
     replanejamento e manobra de recuperacao.

Estrutura:
  * Robo      -> tudo que toca o simulador (sensores, motores, estado).
  * Navegador -> tudo que decide (rota global, alvo local, DWA, recuperacao).
  * main()    -> amarra os dois + visualizacao.
"""

import math

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import dynamic_window_approach as dw
import mapa_ocupacao as mo
from visualizacao import MapaRota


# ============================= CONFIGURACAO =============================
# Se o robo andar "de costas", troque para -1 (inverte 180 graus).
# A frente e detectada automaticamente pelo sensor dianteiro.
SINAL_FRENTE = +1

# Margens de inflacao do A*, da conservadora para a minima (m).
MARGENS_ASTAR = (0.20, 0.16, 0.12)

# Distancia para considerar o objetivo atingido (m).
DIST_GOAL = 0.20

# Raio de vizinhanca dos obstaculos estaticos considerados pelo DWA (m).
RAIO_VIZINHANCA = 1.4

# Passos sem progresso ate disparar replanejamento + recuperacao.
PASSOS_TRAVADO = 30

# Atualiza a janela do mapa a cada N passos (menor = mais fluido/lento).
PASSOS_VISUAL = 5

# Geometria da base diferencial (m).
RAIO_RODA = 0.0375
ENTRE_EIXOS = 0.15


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# ========================================================================
#  ROBO: interface com o CoppeliaSim
# ========================================================================
class Robo:
    def __init__(self, sim):
        self.sim = sim

        self.motor_dir = sim.getObject("/MOTOR_DIREITO")
        self.motor_esq = sim.getObject("/MOTOR_ESQUERDO")
        self.corpo = sim.getObjectParent(self.motor_dir)
        self.goal = sim.getObject("/Goal")
        self.sensores = [
            sim.getObject("/SENSOR_MEIO"),
            sim.getObject("/SENSOR_DIAG_DIREITO"),
            sim.getObject("/SENSOR_DIAG_ESQUERDO"),
            sim.getObject("/SENSOR_DIREITO"),
            sim.getObject("/SENSOR_ESQUERDO"),
        ]

        pos = sim.getObjectPosition(self.corpo, -1)
        ori = sim.getObjectOrientation(self.corpo, -1)
        self.z = pos[2]
        self.roll, self.pitch = ori[0], ori[1]

        # Offset entre o yaw do modelo e a frente FISICA (sensor dianteiro
        # detecta ao longo do +Z local). Corrige o "andar de costas".
        m = sim.getObjectMatrix(self.sensores[0], -1)
        frente_yaw = math.atan2(m[6], m[2])
        flip = 0.0 if SINAL_FRENTE >= 0 else math.pi
        self.heading_offset = normalize_angle(frente_yaw - ori[2] + flip)
        print(f"Heading offset: {math.degrees(self.heading_offset):.1f} graus")

    # ------------------------------------------------------------------
    def estado(self, v=0.0, w=0.0):
        """[x, y, theta_frente, v, w] com theta no rumo fisico da frente."""
        pos = self.sim.getObjectPosition(self.corpo, -1)
        yaw = self.sim.getObjectOrientation(self.corpo, -1)[2]
        theta = normalize_angle(yaw + self.heading_offset)
        return np.array([pos[0], pos[1], theta, v, w], dtype=float)

    def ler_sensores(self):
        """Obstaculos detectados pelos sensores, em coordenadas do mundo."""
        pontos = []
        for sensor in self.sensores:
            res, dist, ponto, _, _ = self.sim.readProximitySensor(sensor)
            if res <= 0:
                continue
            p = np.array(ponto, dtype=float)
            if np.linalg.norm(p) <= 0.0 and dist > 0.0:
                p = np.array([0.0, 0.0, dist])
            m = self.sim.getObjectMatrix(sensor, -1)
            wx = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3]
            wy = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7]
            pontos.append([wx, wy])
        return np.array(pontos, dtype=float) if pontos else np.empty((0, 2))

    # ------------------------------------------------------------------
    def aplicar(self, u, x, dt, bloquear_avanco=False):
        """Aplica [v, w]: integra a pose, comanda as rodas e avanca o passo.

        bloquear_avanco: se True, zera v (paraquedas anti-colisao) mas
        mantem o giro.
        """
        v, w = float(u[0]), float(u[1])
        if bloquear_avanco:
            v = 0.0

        novo = x.copy()
        novo[2] = normalize_angle(x[2] + w * dt)
        novo[0] += v * math.cos(novo[2]) * dt
        novo[1] += v * math.sin(novo[2]) * dt
        novo[3], novo[4] = v, w

        # Cinematica diferencial -> velocidades das rodas
        wr = (2.0 * v + w * ENTRE_EIXOS) / (2.0 * RAIO_RODA)
        wl = (2.0 * v - w * ENTRE_EIXOS) / (2.0 * RAIO_RODA)
        self.sim.setJointTargetVelocity(self.motor_dir, max(min(wr, 20.0), -20.0))
        self.sim.setJointTargetVelocity(self.motor_esq, max(min(wl, 20.0), -20.0))

        # Pose cinematica (robo guiado pelo modelo; rodas dao o visual)
        yaw_modelo = normalize_angle(novo[2] - self.heading_offset)
        self.sim.setObjectPosition(self.corpo, -1, [novo[0], novo[1], self.z])
        self.sim.setObjectOrientation(self.corpo, -1, [self.roll, self.pitch, yaw_modelo])
        try:
            self.sim.resetDynamicObject(self.corpo)
        except Exception:
            pass

        self.sim.step()
        return novo

    def parar(self):
        self.sim.setJointTargetVelocity(self.motor_dir, 0.0)
        self.sim.setJointTargetVelocity(self.motor_esq, 0.0)

    def posicao_goal(self):
        p = self.sim.getObjectPosition(self.goal, -1)
        return [p[0], p[1]]


# ========================================================================
#  NAVEGADOR: rota global + alvo local + DWA + recuperacao
# ========================================================================
class Navegador:
    def __init__(self, obstaculos_estaticos, goal):
        self.dwa = dw.DWAController()
        self.estaticos = obstaculos_estaticos
        self.goal = list(goal)

        self.caminho = [tuple(goal)]
        self.pontos_chave = [tuple(goal)]
        self.indice = 0

        self._stuck_ref = None
        self._stuck = 0
        self._recovery = 0
        self.replanejou = False  # sinaliza para a visualizacao

    # ------------------------------------------------------------------
    def planejar(self, pos):
        """Rota global com margem adaptativa + diagnostico em falha."""
        self.caminho, self.pontos_chave, planner, rr = dw.planejar_rota(
            self.estaticos, (pos[0], pos[1]), self.goal, margens=MARGENS_ASTAR
        )
        self.indice = 0

        if planner.last_plan_failed:
            print("A* NAO encontrou rota; seguindo em linha reta.")
            self._diagnostico(pos)
        else:
            print(f"A*: rota com {len(self.caminho)} pontos, "
                  f"{len(self.pontos_chave)} pontos-chave (margem rr={rr:.2f} m)")

    def _diagnostico(self, pos):
        if len(self.estaticos) == 0:
            return
        d_ini = float(np.min(np.hypot(self.estaticos[:, 0] - pos[0],
                                      self.estaticos[:, 1] - pos[1])))
        d_goal = float(np.min(np.hypot(self.estaticos[:, 0] - self.goal[0],
                                       self.estaticos[:, 1] - self.goal[1])))
        raio = self.dwa.collision_radius
        print(f"   diag: obstaculo mais proximo do INICIO={d_ini*100:.0f} cm, "
              f"do GOAL={d_goal*100:.0f} cm (raio de colisao={raio*100:.0f} cm)")
        if d_goal < raio:
            print("   -> Goal encostado/dentro de obstaculo: afaste-o ~20 cm da parede.")
        if d_ini < raio:
            print("   -> Robo comeca colado a obstaculo: verifique a posicao inicial.")

    # ------------------------------------------------------------------
    def _visivel(self, x, ponto):
        """Linha de visao livre do robo ate o ponto (folga > raio colisao)."""
        if len(self.estaticos) == 0:
            return True
        comprimento = math.hypot(ponto[0] - x[0], ponto[1] - x[1])
        n = max(2, int(comprimento / 0.05) + 1)
        ts = np.linspace(0.0, 1.0, n)
        px = x[0] + (ponto[0] - x[0]) * ts
        py = x[1] + (ponto[1] - x[1]) * ts
        d = np.hypot(px[:, None] - self.estaticos[:, 0],
                     py[:, None] - self.estaticos[:, 1])
        return float(d.min()) > self.dwa.collision_radius + 0.02

    def alvo_local(self, x):
        """Ponto do caminho a perseguir: o mais distante da janela de
        lookahead que ainda esteja na linha de visao do robo (nunca um
        waypoint atras de uma quina)."""
        # Avanca o indice conforme o robo se aproxima
        while (self.indice < len(self.caminho) - 1
               and math.hypot(self.caminho[self.indice][0] - x[0],
                              self.caminho[self.indice][1] - x[1]) < 0.45):
            self.indice += 1

        fim = min(self.indice + 6, len(self.caminho) - 1)
        for k in range(fim, self.indice - 1, -1):
            if self._visivel(x, self.caminho[k]):
                return self.caminho[k]
        return self.caminho[self.indice]

    # ------------------------------------------------------------------
    def obstaculos_locais(self, x, leitura_sensores):
        """Estaticos proximos + deteccoes atuais dos sensores."""
        locais = []
        if len(self.estaticos) > 0:
            d = np.hypot(self.estaticos[:, 0] - x[0], self.estaticos[:, 1] - x[1])
            locais.extend(self.estaticos[d <= RAIO_VIZINHANCA].tolist())
        if len(leitura_sensores) > 0:
            locais.extend(leitura_sensores.tolist())
        return np.array(locais, dtype=float) if locais else np.empty((0, 2))

    def colidiria(self, pos):
        if len(self.estaticos) == 0:
            return False
        d = np.hypot(self.estaticos[:, 0] - pos[0], self.estaticos[:, 1] - pos[1])
        return float(d.min()) <= self.dwa.collision_radius

    # ------------------------------------------------------------------
    def _detectar_travamento(self, x):
        ref = self._stuck_ref
        if ref is None or math.hypot(x[0] - ref[0], x[1] - ref[1]) > 0.06:
            self._stuck_ref = (float(x[0]), float(x[1]))
            self._stuck = 0
        else:
            self._stuck += 1
        return self._stuck >= PASSOS_TRAVADO

    def _recuperar(self, x, alvo):
        """Escape de minimo local: recua, gira para o alvo, avanca."""
        ang = math.atan2(alvo[1] - x[1], alvo[0] - x[0])
        turn = normalize_angle(ang - x[2])
        w = max(min(1.2 * turn, self.dwa.max_yaw_rate), -self.dwa.max_yaw_rate)

        if self._recovery > 22:
            return [-0.08, 0.3 * w]                      # recua
        if abs(turn) > 0.30:
            return [0.0, 0.9 if turn >= 0.0 else -0.9]   # gira no eixo
        return [0.12, 0.5 * w]                           # empurra

    # ------------------------------------------------------------------
    def comando(self, x, leitura_sensores):
        """Decide o comando [v, w] do passo atual."""
        alvo = self.alvo_local(x)
        self.replanejou = False

        if self._recovery > 0:
            self._recovery -= 1
            return self._recuperar(x, alvo)

        obst = self.obstaculos_locais(x, leitura_sensores)
        u, _ = self.dwa.plan(x[0:3], x[3], x[4], alvo, obst)

        if self._detectar_travamento(x):
            print("Robo preso -> replanejando + recuperacao")
            self.planejar(x)
            self.replanejou = True
            self._recovery = 35
            self._stuck = 0

        return u


# ========================================================================
#  PROGRAMA PRINCIPAL
# ========================================================================
def main():
    print("Conectando ao CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.require("sim")

    sim.setStepping(True)
    sim.startSimulation()

    robo = None
    visual = None
    try:
        robo = Robo(sim)
        x = robo.estado()
        goal = robo.posicao_goal()

        # ---- Mapa de Grade de Ocupacao (sensor de visao) ----
        excluir = [
            (x[0], x[1], 0.35),          # nao bloquear o proprio robo
            (goal[0], goal[1], 0.25),    # nem o objetivo
        ]
        try:
            floor = sim.getObject("/Floor")
            estaticos, grade, info = mo.construir_obstaculos_por_visao(
                sim, floor, excluir=excluir
            )
            print(f"Grade de ocupacao: {grade.shape[1]}x{grade.shape[0]} celulas, "
                  f"celula={info['passo']*100:.1f} cm, "
                  f"{len(estaticos)} pontos de obstaculo")
            if len(estaticos) < 4:
                raise RuntimeError("grade vazia")
        except Exception as exc:
            print(f"Grade de ocupacao indisponivel ({exc}); usando bounding boxes.")
            estaticos = mo.construir_obstaculos_por_bbox(sim, robo.corpo)

        # ---- Rota global + visualizacao ----
        nav = Navegador(estaticos, goal)
        nav.planejar(x)

        visual = MapaRota(estaticos, nav.caminho, nav.pontos_chave, x, goal)

        # ---- Loop de navegacao ----
        passo = 0
        while True:
            goal = robo.posicao_goal()          # segue o Goal se ele mover
            nav.goal = goal

            u = nav.comando(x, robo.ler_sensores())
            if nav.replanejou:
                visual.nova_rota(nav.caminho, nav.pontos_chave)

            prox = x.copy()
            prox[2] = normalize_angle(x[2] + u[1] * nav.dwa.dt)
            prox[0] += u[0] * math.cos(prox[2]) * nav.dwa.dt
            prox[1] += u[0] * math.sin(prox[2]) * nav.dwa.dt
            x = robo.aplicar(u, x, nav.dwa.dt,
                             bloquear_avanco=nav.colidiria(prox))

            passo += 1
            if passo % PASSOS_VISUAL == 0:
                visual.atualizar(x)

            dist = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if passo % 10 == 0:
                print(f"step {passo}  dist {dist:.2f}  "
                      f"pos [{x[0]:.2f}, {x[1]:.2f}]  "
                      f"u [{u[0]:.2f}, {u[1]:.2f}]")

            if dist <= DIST_GOAL:
                print("GOAL ATINGIDO!")
                visual.atualizar(x)
                break

    except KeyboardInterrupt:
        print("Parado pelo usuario.")

    finally:
        if robo is not None:
            robo.parar()
        sim.stopSimulation()
        print("Simulacao encerrada.")
        if visual is not None:
            visual.salvar("mapa_rota.png")
            visual.fechar(manter_aberto=True)


if __name__ == "__main__":
    main()
