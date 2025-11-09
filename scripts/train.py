# scripts/train.py
from __future__ import annotations
import argparse
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from agents.utils import set_seed, ensure_dir
from envs.racing_env import RacingEnv
from agents.dqn_agent import crear_dqn

class RenderPreviewCallback(BaseCallback):
    def __init__(self, ruta_csv: str, every_n_steps: int = 5000,
                 ppu: int = 36, fps: int = 24, speed_scale: float = 0.5):
        super().__init__()
        self.ruta_csv = ruta_csv
        self.every_n_steps = max(1, every_n_steps)
        self.ppu = ppu
        self.fps = fps
        self.speed_scale = speed_scale

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            # Entorno separado SOLO para previsualizar
            env = RacingEnv(
                ruta_csv=self.ruta_csv, patch_h=11, patch_w=11,
                render_mode="human", renderer_ppu=self.ppu, render_fps=self.fps
            )
            env.set_visual_speed_scale(self.speed_scale)
            obs, info = env.reset()
            done, trunc = False, False
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, _ = env.step(int(action))
            env.close()
        return True

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento DQN para pista CSV")
    parser.add_argument("--csv", type=str, default="tracks/track01_recta.csv", help="Ruta a la pista CSV")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Pasos totales de entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=0, help="Si >0, renderiza cada N steps (costo alto)")
    parser.add_argument("--modelo-out", type=str, default="models/dqn_track01.zip")
    parser.add_argument("--preview-every", type=int, default=0, help="Cada N steps, previsualiza 1 episodio renderizado")
    parser.add_argument("--preview-ppu", type=int, default=36)
    parser.add_argument("--preview-fps", type=int, default=24)
    parser.add_argument("--preview-speed", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)
    env = RacingEnv(ruta_csv=args.csv, patch_h=11, patch_w=11, render_mode=None)
    env = Monitor(env)

    model = crear_dqn(env)

    callbacks = None
    if args.preview_every > 0:
        callbacks = RenderPreviewCallback(
            ruta_csv=args.csv,
            every_n_steps=args.preview_every,
            ppu=args.preview_ppu,
            fps=args.preview_fps,
            speed_scale=args.preview_speed
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    ensure_dir("models")
    model.save(args.modelo_out)
    print(f"Modelo guardado en: {args.modelo_out}")

if __name__ == "__main__":
    main()