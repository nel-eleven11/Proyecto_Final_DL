# scripts/visualize.py
from __future__ import annotations
import argparse
from stable_baselines3 import DQN
from envs.racing_env import RacingEnv

def main():
    parser = argparse.ArgumentParser(description="Reproducir episodios con un modelo DQN entrenado")
    parser.add_argument("--csv", type=str, default="tracks/track01.csv")
    parser.add_argument("--modelo", type=str, default="models/dqn_track01.zip")
    parser.add_argument("--episodios", type=int, default=3)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--ppu", type=int, default=36, help="Píxeles por unidad en renderer")
    parser.add_argument("--fps", type=int, default=30, help="FPS de visualización")
    parser.add_argument("--speed-scale", type=float, default=0.4, help="Escala de velocidad SOLO visual (0.1..1.0)")

    args = parser.parse_args()

    env = RacingEnv(
        ruta_csv=args.csv,
        patch_h=13,
        patch_w=13,
        render_mode=("human" if args.render else None),
        renderer_ppu=args.ppu,
        render_fps=args.fps
    )

    env.set_visual_speed_scale(args.speed_scale)
    env.set_render_fps(args.fps)
    model = DQN.load(args.modelo, env=env)

    for ep in range(args.episodios):
        obs, info = env.reset()
        terminado, trunc = False, False
        R = 0.0
        while not (terminado or trunc):
            accion, _ = model.predict(obs, deterministic=True)
            obs, r, terminado, trunc, info = env.step(int(accion))
            R += r
        print(f"Episodio {ep+1}: retorno = {R:.2f}, meta={info.get('meta')}, choque={info.get('choque')}")
    env.close()

if __name__ == "__main__":
    main()