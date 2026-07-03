"""
Suite de testes sinteticos (sem CoppeliaSim).

Valida a logica REAL de navegacao (classe Navegador do main.py) em malha
fechada, nos casos criticos:

  1. Corredores estreitos (0.60 / 0.50 / 0.40 / 0.35 m)
  2. Obstaculo em L (robo comeca "dentro" da quina)
  3. Goal a 15 cm da parede
  4. Vao apertado curto vs corredor largo -> deve escolher o LARGO
  5. So existe o vao apertado -> deve usa-lo mesmo assim
  6. Robo comecando virado para o lado errado (180 graus)

Uso:  python utils/testar_algoritmos.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dynamic_window_approach as dw
from main import Navegador, normalize_angle


# ------------------------------------------------------------------
def arena(lado=4.0, passo=0.1):
    pontos = []
    v = 0.0
    while v <= lado:
        pontos += [[v, 0.0], [v, lado], [0.0, v], [lado, v]]
        v += passo
    return pontos


def parede_vertical(pontos, x, lado=4.0, vaos=(), passo=0.05):
    """Parede em x= const com aberturas (y0, y1) listadas em vaos."""
    y = 0.0
    while y <= lado:
        if not any(a < y < b for a, b in vaos):
            pontos.append([x, y])
        y += passo
    return pontos


# ------------------------------------------------------------------
def navegar(pontos, inicio, goal, theta0=0.0, max_passos=2000):
    """Malha fechada com o Navegador real. Retorna (ok, passos, folga_min, x)."""
    obst = np.array(pontos, dtype=float)
    nav = Navegador(obst, list(goal))
    nav.planejar(np.array([inicio[0], inicio[1], theta0, 0.0, 0.0]))

    x = np.array([inicio[0], inicio[1], theta0, 0.0, 0.0])
    folga = float("inf")

    for passo in range(max_passos):
        u = nav.comando(x, np.empty((0, 2)))

        # Integra a cinematica (papel do CoppeliaSim na simulacao real)
        x[2] = normalize_angle(x[2] + u[1] * nav.dwa.dt)
        x[0] += u[0] * math.cos(x[2]) * nav.dwa.dt
        x[1] += u[0] * math.sin(x[2]) * nav.dwa.dt
        x[3], x[4] = u

        d = float(np.min(np.hypot(obst[:, 0] - x[0], obst[:, 1] - x[1])))
        folga = min(folga, d)
        if d < 0.11:
            return False, passo, folga, x          # colidiu
        if math.hypot(goal[0] - x[0], goal[1] - x[1]) <= 0.20:
            return True, passo, folga, x           # chegou

    return False, max_passos, folga, x             # nao chegou


def cruzamento_y(caminho, x_parede=2.0):
    for (x0, y0), (x1, y1) in zip(caminho, caminho[1:]):
        if (x0 - x_parede) * (x1 - x_parede) <= 0 and x0 != x1:
            t = (x_parede - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


# ------------------------------------------------------------------
def rodar_suite():
    resultados = []

    def caso(nome, ok, extra=""):
        status = "PASSOU" if ok else "FALHOU"
        resultados.append(ok)
        print(f"[{status}] {nome} {extra}")

    # 1) corredores estreitos
    for L in (0.60, 0.50, 0.40, 0.35):
        pts = arena()
        pts = parede_vertical(pts, 2.0, vaos=[(2.0 - L / 2, 2.0 + L / 2)])
        ok, n, folga, _ = navegar(pts, (0.6, 2.0), (3.4, 2.0))
        caso(f"corredor {L:.2f} m", ok, f"({n} passos, folga {folga:.2f} m)")

    # 2) obstaculo em L (robo dentro da quina)
    pts = arena()
    t = 0.0
    while t <= 1.2:
        pts.append([1.5 + t, 2.2])
        pts.append([2.7, max(2.2 - t, 1.0)])
        t += 0.05
    ok, n, folga, _ = navegar(pts, (1.9, 2.7), (3.3, 1.4))
    caso("obstaculo em L (quina)", ok, f"({n} passos, folga {folga:.2f} m)")

    # 3) goal a 15 cm da parede
    pts = arena()
    ok, n, folga, _ = navegar(pts, (0.6, 2.0), (3.85, 2.0))
    caso("goal a 15 cm da parede", ok, f"({n} passos, folga {folga:.2f} m)")

    # 4) vao apertado curto (0.30 m) vs corredor largo (0.70 m) -> escolher o largo
    pts = arena()
    pts = parede_vertical(pts, 2.0, vaos=[(3.35, 3.65), (2.05, 2.75)])
    caminho, _, _, _ = dw.planejar_rota(
        np.array(pts), (2.8, 3.5), (1.2, 3.5)
    )
    y = cruzamento_y(caminho)
    caso("prefere corredor largo ao vao apertado",
         y is not None and 2.0 < y < 2.8, f"(cruza parede em y={y:.2f})")

    # 5) SO existe o vao apertado (0.45 m) -> deve usa-lo
    pts = arena()
    pts = parede_vertical(pts, 2.0, vaos=[(3.28, 3.73)])
    ok, n, folga, _ = navegar(pts, (2.8, 3.5), (1.2, 3.5))
    caso("vao apertado como unica opcao", ok, f"({n} passos, folga {folga:.2f} m)")

    # 6) robo comecando virado 180 graus para o lado errado
    pts = arena()
    pts = parede_vertical(pts, 2.0, vaos=[(1.7, 2.3)])
    ok, n, folga, _ = navegar(pts, (0.6, 2.0), (3.4, 2.0), theta0=math.pi)
    caso("inicio virado 180 graus", ok, f"({n} passos, folga {folga:.2f} m)")

    print()
    total, passaram = len(resultados), sum(resultados)
    print(f"===== {passaram}/{total} casos passaram =====")
    return passaram == total


if __name__ == "__main__":
    sucesso = rodar_suite()
    sys.exit(0 if sucesso else 1)
