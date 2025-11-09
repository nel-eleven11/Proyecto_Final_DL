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

# Entrenar al agente
python scripts/train.py   --csv tracks/TRACK.csv   --timesteps N  --modelo-out models/MODELO.zip

#Visualizar al agente ya entrenado
python scripts/visualize.py --csv tracks/TRACK.csv --modelo models/MODELO1.zip --render True
```