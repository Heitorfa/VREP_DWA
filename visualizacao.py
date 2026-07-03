"""
Visualizacao do Mapa de Ocupacao + Rota do Robo.

Mostra, a cada simulacao:
  * os pontos de obstaculo (grade de ocupacao),
  * a rota global planejada pelo A* melhorado,
  * os pontos-chave (metas do DWA),
  * o robo, o objetivo e a trajetoria real percorrida.

A janela abre no inicio da navegacao e e atualizada ao vivo; ao final, a
figura e salva em PNG (mapa_rota.png ao lado do main.py). Em ambiente sem
interface grafica, cai automaticamente no modo "apenas salvar".
"""

import matplotlib

try:  # tenta backend interativo; sem GUI, usa Agg (apenas salvar PNG)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.close()
    INTERATIVO = matplotlib.get_backend().lower() != "agg"
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INTERATIVO = False


class MapaRota:
    """Figura ao vivo com mapa de ocupacao, rota planejada e trajetoria real."""

    def __init__(self, obstaculos, caminho, pontos_chave, inicio, goal,
                 titulo="Mapa de Ocupação + Rota do Robô"):
        if INTERATIVO:
            plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5))
        self.ax.set_title(titulo)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.25)

        # Obstaculos (grade de ocupacao)
        if obstaculos is not None and len(obstaculos) > 0:
            self.ax.plot(
                obstaculos[:, 0], obstaculos[:, 1],
                ".", color="black", markersize=2, label="Obstáculos",
            )

        # Rota planejada + pontos-chave (atualizaveis em replanejamento)
        self.linha_rota, = self.ax.plot(
            [p[0] for p in caminho], [p[1] for p in caminho],
            "-", color="#1464dc", linewidth=2, label="Rota planejada A*",
        )
        self.linha_chaves, = self.ax.plot(
            [p[0] for p in pontos_chave], [p[1] for p in pontos_chave],
            "x", color="#b8a000", markersize=9, mew=2, label="Pontos-chave",
        )

        # Robo (posicao atual), goal e trajetoria real
        self.ax.plot(inicio[0], inicio[1], "o", color="green",
                     markersize=10, label="Robô")
        self.ax.plot(goal[0], goal[1], "o", color="red",
                     markersize=11, label="Goal")
        self.traj_x, self.traj_y = [float(inicio[0])], [float(inicio[1])]
        self.linha_traj, = self.ax.plot(
            self.traj_x, self.traj_y,
            "-", color="#c832c8", linewidth=2.2, label="Trajetória real",
        )
        self.ponto_robo, = self.ax.plot(
            [inicio[0]], [inicio[1]], "o", color="#00a000", markersize=8,
        )

        self.ax.legend(loc="best", fontsize=9)

        if INTERATIVO:
            self.fig.show()
            self._desenhar()

    # ------------------------------------------------------------------
    def _desenhar(self):
        if INTERATIVO:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass

    def atualizar(self, pos):
        """Acrescenta a posicao atual do robo a trajetoria real."""
        self.traj_x.append(float(pos[0]))
        self.traj_y.append(float(pos[1]))
        self.linha_traj.set_data(self.traj_x, self.traj_y)
        self.ponto_robo.set_data([pos[0]], [pos[1]])
        self._desenhar()

    def nova_rota(self, caminho, pontos_chave):
        """Atualiza a rota exibida apos um replanejamento."""
        self.linha_rota.set_data(
            [p[0] for p in caminho], [p[1] for p in caminho]
        )
        self.linha_chaves.set_data(
            [p[0] for p in pontos_chave], [p[1] for p in pontos_chave]
        )
        self._desenhar()

    def salvar(self, arquivo="mapa_rota.png"):
        try:
            self.fig.savefig(arquivo, dpi=130, bbox_inches="tight")
            print(f"Mapa salvo em {arquivo}")
        except Exception as exc:
            print("Nao foi possivel salvar o mapa:", exc)

    def fechar(self, manter_aberto=True):
        """Ao final: mantem a janela na tela (bloqueante) ou fecha."""
        if INTERATIVO and manter_aberto:
            try:
                plt.ioff()
                plt.show()
            except Exception:
                pass
        else:
            plt.close(self.fig)
