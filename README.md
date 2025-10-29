# Proyecto_Final_DL

# RL Racer (DQN)

Este proyecto implementa un entorno tipo pista en cuadriculado (CSV) y un agente **DQN** con Stable-Baselines3.
Está adaptado a las definiciones solicitadas: solo hay choques con muros, superficies afectan velocidad (aceite, terracería, boost), y se usa observación **patch egocéntrico** para DQN.

## Estructura
```
rl_racer/
 ├─ envs/
 │   ├─ grid_track.py
 │   ├─ dynamics.py
 │   ├─ sensors.py
 │   ├─ rewards.py
 │   ├─ renderer.py
 │   └─ racing_env.py
 ├─ agents/
 │   ├─ dqn_agent.py
 │   └─ utils.py
 ├─ configs/
 │   └─ dqn.yaml
 ├─ tracks/
 │   └─ track01_recta.csv
 ├─ scripts/
 │   ├─ train.py
 │   └─ visualize.py
 └─ requirements.txt
```

## Ejecución rápida
```bash
pip install -r requirements.txt
python scripts/train.py --timesteps 200000 --render-every 0
# Luego visualizar
python scripts/visualize.py --modelo models/dqn_track01.zip --episodios 5 --render True
```

> Nota: Por simplicidad inicial, el coche está orientado hacia la derecha (eje X creciente) y el **patch egocéntrico** mira hacia adelante. 
Dimensiones del coche en el grid: **largo_x = 2**, **alto_y = 4** unidades (ajustable en `racing_env.py`). 
Si quieres 4×6 u otras dimensiones, cambia `CAR_LARGO_X` y `CAR_ALTO_Y` y el `PIXELS_POR_UNIDAD` en el renderer.