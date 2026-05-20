# Navegação Autônoma de Robôs Móveis com A* e DWA no CoppeliaSim

Este repositório contém a implementação de um sistema híbrido de navegação autônoma para robôs móveis operando em ambiente simulado no CoppeliaSim. A arquitetura desenvolvida integra um planejador global de caminhos baseado no algoritmo **A*** com um controlador reativo local fundamentado na **Abordagem de Janela Dinâmica (Dynamic Window Approach - DWA)**. A comunicação e o controle do simulador são realizados em tempo real utilizando a **ZeroMQ Remote API** do CoppeliaSim.

## 🚀 Funcionalidades

- **Mapeamento e Planejamento Global (A*):** Varre a árvore de objetos do cenário para identificar obstáculos estáticos, inflar margens de segurança com base no raio do robô e gerar uma malha de waypoints otimizada até o objetivo final.
- **Navegação Reativa Local (DWA):** Amostra o espaço de velocidades lineares e angulares dentro das restrições dinâmicas do robô, prevendo trajetórias e minimizando uma função de custo multiobjetivo.
- **Fusão de Sensores em Tempo Real:** Combina a geometria de obstáculos estáticos mapeados com leituras dinâmicas de sensores de proximidade ultrassônicos para evasão de colisões imprevistas.
- **Salvaguarda Mecânica (Controle de Emergência):** Em cenários de bloqueio ou alto custo de colisão, o controlador assume uma manobra de rotação forçada no próprio eixo para reorientar a base móvel.

## 📁 Estrutura do Projeto

O sistema é dividido em dois módulos principais interdependentes:

1. **`script_mirror.py` (Módulo de Integração e Simulação):** - Gerencia o ciclo de vida da simulação no CoppeliaSim.
   - Realiza a varredura do ambiente para extração de obstáculos estáticos e leitura dos sensores de proximidade.
   - Implementa o loop principal de controle e faz o rastreamento de progresso ao longo dos waypoints gerados pelo A*.
2. **`dynamic_window_approach.py` (Módulo Algorítmico):**
   - Contém a classe `AStarPlanner` para o cálculo da rota global ideal.
   - Contém a classe `DWAController` responsável pelo cálculo do espaço vetorial de comandos válidos $[v, \omega]$, simulação de trajetórias dinâmicas futuras e minimização das funções de custo.

## 🛠️ Pré-requisitos

Antes de executar, certifique-se de possuir as seguintes ferramentas instaladas:

- Python 3.8 ou superior
- NumPy
- CoppeliaSim (versão moderna compatível com ZeroMQ Remote API)

Instale as dependências via pip:
```bash
pip install numpy coppeliasim-zmqremoteapi-client