# scripts/train.py
from __future__ import annotations
import argparse
import time 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from agents.utils import set_seed, ensure_dir
from envs.racing_env import RacingEnv
from agents.dqn_agent import crear_dqn

class RenderPreviewCallback(BaseCallback):
    def __init__(self, ruta_csv: str, every_n_steps: int = 5000,
                 ppu: int = 36, fps: int = 24, speed_scale: float = 0.7):
        super().__init__()
        self.ruta_csv = ruta_csv
        self.every_n_steps = max(1, every_n_steps)
        self.ppu = ppu
        self.fps = fps
        self.speed_scale = speed_scale

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            env = RacingEnv(ruta_csv=args.csv, patch_h=13, patch_w=13, render_mode=None)
            env.set_visual_speed_scale(self.speed_scale)
            obs, _ = env.reset()
            done, trunc = False, False
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
            env.close()
        return True

class StatsCallback(BaseCallback):
    """Acumula métricas. Si print_per_episode=True imprime por episodio; si no, solo resumen final."""
    def __init__(self, print_per_episode: bool = True):
        super().__init__()
        self.print_per_episode = print_per_episode
        self.ep_returns = []
        self.ep_lengths = []
        self.successes = 0
        self.crashes = 0
        self.start_time: float | None = None
        self.end_time: float | None = None

    def _on_training_start(self) -> None:
        # Guardar hora de inicio de entrenamiento
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for d, info in zip(dones, infos):
            if d:
                ep = info.get("episode")
                if ep:
                    self.ep_returns.append(float(ep.get("r", 0.0)))
                    self.ep_lengths.append(int(ep.get("l", 0)))
                if info.get("meta"):
                    self.successes += 1
                if info.get("choque"):
                    self.crashes += 1
                if self.print_per_episode:
                    idx = len(self.ep_returns)
                    print(f"[EP {idx}] R={self.ep_returns[-1]:.2f} | L={self.ep_lengths[-1]} | meta={bool(info.get('meta'))} | choque={bool(info.get('choque'))}")
        return True

    def _on_training_end(self) -> None:
        if self.start_time is not None:
            self.end_time = time.time()
        if not self.ep_returns:
            print("No se registraron episodios (¿timesteps muy bajos?).")
            if self.start_time is not None:
                dur = self.end_time - self.start_time if self.end_time else 0.0
                print(f"Tiempo total de entrenamiento: {dur:.2f} s")
            return

        import numpy as np
        R = np.array(self.ep_returns, dtype=float)
        L = np.array(self.ep_lengths, dtype=int)
        dur = 0.0
        if self.start_time is not None and self.end_time is not None:
            dur = self.end_time - self.start_time

        resumen = {
            "episodios": int(len(R)),
            "retorno_prom": float(R.mean()),
            "retorno_std": float(R.std()),
            "retorno_mejor": float(R.max()),
            "largo_prom": float(L.mean()),
            "exitos": int(self.successes),
            "choques": int(self.crashes),
            "tasa_exito": float(self.successes / len(R)),
            "tiempo_total": float(dur),
        }
        print("\n=== Resumen de entrenamiento ===")
        for k, v in resumen.items():
            print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento DQN para pista CSV")
    parser.add_argument("--csv", type=str, default="tracks/track01.csv", help="Ruta a la pista CSV")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Pasos totales de entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo-out", type=str, default="models/dqn_track01.zip")
    parser.add_argument("--preview-every", type=int, default=0, help="Cada N steps, previsualiza 1 episodio renderizado")
    parser.add_argument("--preview-ppu", type=int, default=36)
    parser.add_argument("--preview-fps", type=int, default=24)
    parser.add_argument("--preview-speed", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    env = RacingEnv(ruta_csv=args.csv, patch_h=13, patch_w=13, render_mode=None)
    env = Monitor(env)

    # Si hay muchos timesteps, reducimos verbosidad y solo mostramos RESUMEN final
    print_per_ep = args.timesteps <= 5000
    verbose_agent = 1 if print_per_ep else 0

    model = crear_dqn(env, verbose=verbose_agent)

    stats_cb = StatsCallback(print_per_episode=print_per_ep)
    callbacks = [stats_cb]
    if args.preview_every > 0:
        callbacks.append(RenderPreviewCallback(
            ruta_csv=args.csv,
            every_n_steps=args.preview_every,
            ppu=args.preview_ppu,
            fps=args.preview_fps,
            speed_scale=args.preview_speed
        ))

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))

    ensure_dir("models")
    model.save(args.modelo_out)
    print(f"\nModelo guardado en: {args.modelo_out}")

if __name__ == "__main__":
    main()