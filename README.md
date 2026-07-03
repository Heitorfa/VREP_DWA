# Navegação Autônoma de Robôs Móveis — A* Melhorado + DWA + Grade de Ocupação (CoppeliaSim)

Sistema híbrido de navegação autônoma para uma base móvel diferencial simulada no **CoppeliaSim**,
controlada por Python via **ZeroMQ Remote API**. Integra:

- **Planejador global A\* melhorado** (Guo et al., 2024): heurística ponderada `f(n)=g(n)+(1+d/D)·h(n)`,
  seleção de pontos-chave (remoção de colineares + simplificação por linha de visão) e suavização por
  **curvas de Bézier de 2ª ordem**.
- **Mapa de Grade de Ocupação** construído por um **sensor de visão** criado por código, apontado para
  baixo sobre a cena (modelo de Moravec & Elfes, 1985). É a **fonte primária de obstáculos** do A*.
- **Controlador local DWA** (Dynamic Window Approach) com fusão guiada pelos pontos-chave do A* e camada
  reativa por sensores de proximidade para obstáculos dinâmicos.

Autor: **Heitor Freire Alves** — Disciplina de Robótica, UNIVASF.

A cada simulação, uma janela exibe o **Mapa de Ocupação + Rota do Robô** (obstáculos, rota A*,
pontos-chave, robô, goal e trajetória real), atualizada ao vivo e salva em `mapa_rota.png` ao final.

## 📁 Estrutura

```
main.py                     # Robo (CoppeliaSim) + Navegador (A*+DWA) + loop principal
dynamic_window_approach.py  # A* melhorado (heurística ponderada, pontos-chave, Bézier,
                            #   penalidade de proximidade) + DWAController + planejar_rota
mapa_ocupacao.py            # Sensor de visão + Mapa de Grade de Ocupação (+ fallback bbox)
visualizacao.py             # Janela ao vivo: mapa de ocupação + rota + trajetória real
utils/
  testar_algoritmos.py      # Suíte de testes sintéticos (9 casos críticos, sem CoppeliaSim)
  testar_conexao.py         # Testa a conexão com o CoppeliaSim
  listar_objetos.py         # Lista os objetos da cena
cenas/
  scena com obstaculos.ttt  # Cena oficial
docs/
  Relatorio_Final_Navegacao_Autonoma.pdf   # Relatório final
  Apresentacao_Navegacao_Autonoma.pptx     # Apresentação
  artigo_Guo2024_improved_Astar_DWA.pdf    # Artigo de referência
  figuras/                                 # Figuras geradas para o relatório
legado/                     # Arquivos antigos preservados (não usados)
```

## 🛠️ Pré-requisitos

- Python 3.8+
- CoppeliaSim (compatível com a ZeroMQ Remote API)

Instale as dependências:
```bash
pip install numpy coppeliasim-zmqremoteapi-client
# Opcional, apenas para regenerar relatório/figuras/apresentação:
pip install reportlab matplotlib python-pptx pillow
```

## ▶️ Como executar

1. Abra `cenas/scena com obstaculos.ttt` no CoppeliaSim.
2. (Opcional) Teste a conexão: `python utils/testar_conexao.py`
3. (Opcional) Visualize a grade de ocupação isoladamente: `python mapa_ocupacao.py`
4. (Opcional) Rode a suíte de testes dos algoritmos: `python utils/testar_algoritmos.py`
5. Rode a navegação: `python main.py`

O script inicia a simulação em modo *stepping*, constrói a grade de ocupação, planeja a rota com o A*
melhorado e navega até o `/Goal` com o DWA.

## ⚙️ Ajustes úteis

- **Se o robô andar "de costas":** troque `SINAL_FRENTE = +1` para `-1` no topo de `main.py`
  (a frente é detectada automaticamente pelo sensor dianteiro, mas isso inverte 180° se necessário).
- **Grade de ocupação:** parâmetros em `mapa_ocupacao.py` → classe `ConfigGrade`
  (`resolucao`, `altura`, `altura_min_obstaculo`, `flip_x`, `flip_y`). Se a grade sair espelhada em
  relação à cena, ajuste `flip_x` / `flip_y`.
- **Comportamento do DWA:** ganhos em `DWAController.__init__` (`dynamic_window_approach.py`).

## 📚 Referências

- GUO, H. et al. *Path planning of greenhouse electric crawler tractor based on the improved A\* and DWA
  algorithms.* Computers and Electronics in Agriculture, v. 227, 109596, 2024.
- MORAVEC, H.; ELFES, A. *High resolution maps from wide angle sonar.* Proc. 1985 IEEE ICRA, p. 116-121.
- FOX, D.; BURGARD, W.; THRUN, S. *The Dynamic Window Approach to Collision Avoidance.* IEEE R&A Mag., 1997.
