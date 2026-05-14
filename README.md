# Dynamic Window Approach aplicado ao Pioneer P3DX no CoppeliaSim

Projeto desenvolvido para simulação de navegação autônoma utilizando o algoritmo **Dynamic Window Approach (DWA)** integrado ao **CoppeliaSim** através de Python.

O objetivo principal do projeto foi implementar um sistema de desvio de obstáculos em tempo real utilizando uma base móvel já disponibilizada na cena do simulador.

---

# Objetivo

Implementar e testar um algoritmo de navegação local baseado em DWA capaz de:

- mover o robô até um alvo;
- evitar obstáculos de forma autônoma;
- gerar trajetórias em tempo real;
- utilizar sensores ultrassônicos da plataforma Pioneer;
- integrar Python com o CoppeliaSim.

Além disso, foram realizadas modificações no algoritmo original para melhorar o comportamento do robô em ambientes mais fechados e reduzir problemas de travamento em obstáculos.

---

# Tecnologias Utilizadas

- Python 3
- CoppeliaSim
- NumPy
- Matplotlib
- ZeroMQ Remote API

---

# Estrutura do Projeto

```text
.
├── lib/
│   ├── dynamic_window_approach.py
│   └── script_mirror.py
│
├── scenes/
│   └── cena.ttt
│
└── README.md
```

---

# Funcionamento do Sistema

O sistema funciona da seguinte forma:

1. O CoppeliaSim executa a cena contendo:
   - Pioneer P3DX;
   - sensores ultrassônicos;
   - obstáculos;
   - alvo.

2. O script Python conecta-se ao simulador utilizando a API remota.

3. Os sensores do robô detectam obstáculos em tempo real.

4. O algoritmo DWA:
   - prevê várias trajetórias possíveis;
   - calcula custos;
   - escolhe o melhor movimento;
   - envia velocidades para as rodas.

5. O robô navega até o alvo desviando dos obstáculos.

---

# Melhorias Implementadas no DWA

O algoritmo original foi modificado para melhorar o comportamento do robô na simulação.

Entre as alterações realizadas:

- suavização do custo de obstáculos;
- redução de oscilações;
- prevenção de rotação infinita;
- escape de mínimos locais;
- ajuste dinâmico de pesos;
- redução automática de velocidade próximo a obstáculos;
- comportamento inspirado em robôs aspiradores.

Essas modificações tornaram a navegação mais estável e natural dentro do ambiente simulado.

---

# Como Executar

## 1. Abrir o CoppeliaSim

Carregue a cena `.ttt` contendo:
- Pioneer P3DX;
- obstáculos;
- objeto Target.

---

## 2. Iniciar a simulação

Clique no botão **Play** dentro do CoppeliaSim.

---

## 3. Executar o Python

No terminal:

```bash
python script_mirror.py
```

---

# Sensores Utilizados

O robô utiliza sensores ultrassônicos do próprio Pioneer P3DX para:

- detectar obstáculos;
- estimar distâncias;
- auxiliar no cálculo das trajetórias.

---

# Resultados

O sistema foi capaz de:

- navegar autonomamente;
- evitar obstáculos;
- recalcular trajetórias em tempo real;
- alcançar o alvo em diferentes cenários.

Durante os testes foram observados alguns problemas típicos de algoritmos reativos, como mínimos locais, que foram parcialmente corrigidos através de ajustes no DWA.

---

# Possíveis Melhorias Futuras

- utilização de SLAM;
- criação de mapa do ambiente;
- planejamento global;
- integração com ROS;
- uso de LiDAR;
- navegação multiobjetivo;
- otimização do cálculo de custo.

---

# Referências

- Atsushi Sakai — Python Robotics
- Documentação oficial do CoppeliaSim
- Dynamic Window Approach for Mobile Robots

---

# Autor

Projeto desenvolvido para fins acadêmicos na disciplina de Robótica.
