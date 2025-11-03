# scripts/train.py
from __future__ import annotations
import argparse
from stable_baselines3.common.monitor import Monitor
from agents.utils import set_seed, ensure_dir
from envs.racing_env import RacingEnv
from agents.dqn_agent import crear_dqn

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento DQN para pista CSV")
    parser.add_argument("--csv", type=str, default="tracks/track01_recta.csv", help="Ruta a la pista CSV")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Pasos totales de entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=0, help="Si >0, renderiza cada N steps (costo alto)")
    parser.add_argument("--modelo-out", type=str, default="models/dqn_track01.zip")
    args = parser.parse_args()

    set_seed(args.seed)
    env = RacingEnv(ruta_csv=args.csv, patch_h=11, patch_w=11, render_mode=None)
    env = Monitor(env)

    model = crear_dqn(env)

    # Opción de render ocasional (cuidado: lento). Aquí lo omitimos en learn()
    model.learn(total_timesteps=args.timesteps)

    ensure_dir("models")
    model.save(args.modelo_out)
    print(f"Modelo guardado en: {args.modelo_out}")

if __name__ == "__main__":
    main()