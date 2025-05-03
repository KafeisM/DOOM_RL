# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import ReplayMemory, get_epsilon
from model import DQN
from env_wrappers import RewardShapingWrapper, PreprocessFrame, FrameStack
from DoomGame import DoomEnv

def train(resume_path=None, total_steps=3_000_000):
    # --- Configuración fija ---
    MODELS_DIR    = "models"
    LOG_DIR       = "runs/doom_rl"
    SAVE_FREQ     = 50
    BATCH_SIZE    = 64
    LR            = 1e-4
    GAMMA         = 0.99
    TARGET_UPDATE = 1000

    os.makedirs(MODELS_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # --- Crear entorno y wrappers ---
    env = DoomEnv("deathmatch.cfg", frame_skip=4)
    env = RewardShapingWrapper(env)
    env = PreprocessFrame(env, 108, 60, grayscale=False)
    env = FrameStack(env, 4)

    obs_shape = env.observation_space.shape        # (60,108,12)
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    n_actions   = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Crear redes ---
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    memory     = ReplayMemory(100000)

    start_step    = 0
    start_episode = 1
    best_reward   = -float("inf")

    # --- Cargar checkpoint si lo indicamos ---
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        policy_net.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        start_step    = ckpt.get('global_step', 0)
        start_episode = ckpt.get('episode', 1)
        best_reward   = ckpt.get('best_reward', best_reward)
        print(f"Resumed from {resume_path}: step={start_step}, episode={start_episode}, best_reward={best_reward:.2f}")

    # sincronizar target
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    global_step = start_step
    episode = start_episode

    # --- Bucle hasta total_steps ---
    while global_step < total_steps:
        state, _  = env.reset()
        done      = False
        ep_reward = 0.0
        ep_loss   = 0.0
        steps     = 0

        while not done and global_step < total_steps:
            epsilon = get_epsilon(global_step)
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                st = torch.tensor(state, dtype=torch.float32)\
                         .permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad():
                    qv = policy_net(st)
                action = int(qv.argmax(1).item())

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # almacenar
            memory.push((state, action, reward, next_state, done))
            state      = next_state
            ep_reward += reward
            global_step += 1
            steps     += 1

            # actualizar red
            if len(memory) >= BATCH_SIZE:
                s,a,r,ns,done_b = memory.sample(BATCH_SIZE)
                sb = torch.tensor(np.array(s), dtype=torch.float32).permute(0,3,1,2).to(device)
                nsb= torch.tensor(np.array(ns),dtype=torch.float32).permute(0,3,1,2).to(device)
                ab = torch.tensor(a, dtype=torch.int64).to(device)
                rb = torch.tensor(r, dtype=torch.float32).to(device)
                db = torch.tensor(done_b, dtype=torch.float32).to(device)

                with torch.no_grad():
                    next_q = target_net(nsb).max(1)[0]
                    target = rb + GAMMA * next_q * (1 - db)
                q_vals = policy_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss   = F.mse_loss(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

                if global_step % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        # --- Fin episodio ---
        avg_loss = ep_loss / max(steps,1)
        writer.add_scalar("Reward/episode", ep_reward, episode)
        writer.add_scalar("Loss/episode",   avg_loss,   episode)
        writer.add_scalar("Epsilon/episode",epsilon,   episode)

        print(f"Ep {episode:03d} | Step {global_step:,} | Reward {ep_reward:.2f} | Loss {avg_loss:.4f} | ε {epsilon:.3f}")

        # guardar mejor
        if ep_reward > best_reward:
            best_reward = ep_reward
            ckpt = {
                'episode': episode,
                'global_step': global_step,
                'model_state': policy_net.state_dict(),
                'opt_state':   optimizer.state_dict(),
                'best_reward': best_reward
            }
            torch.save(ckpt, os.path.join(MODELS_DIR, "best.pth"))
            print(f"  ▶ New best: {best_reward:.2f}")

        # checkpoint por episodio
        if episode % SAVE_FREQ == 0:
            ckpt = {
                'episode': episode,
                'global_step': global_step,
                'model_state': policy_net.state_dict(),
                'opt_state':   optimizer.state_dict(),
                'best_reward': best_reward
            }
            path = os.path.join(MODELS_DIR, f"ep{episode}.pth")
            torch.save(ckpt, path)
            print(f"  • Checkpoint saved at {path}")

        episode += 1

    # --- Guardar final ---
    ckpt = {
        'episode': episode-1,
        'global_step': global_step,
        'model_state': policy_net.state_dict(),
        'opt_state':   optimizer.state_dict(),
        'best_reward': best_reward
    }
    torch.save(ckpt, os.path.join(MODELS_DIR, "final.pth"))
    print(f"Training finished at step {global_step:,}. Final model saved.")

    writer.close()
    env.close()

if __name__ == "__main__":
    # Para arrancar desde cero:
    #    python train.py
    # Para reanudar de un checkpoint:
    #    python train.py --resume models/best.pth
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a checkpoint para reanudar training")
    parser.add_argument("--steps",  type=int, default=3_000_000,
                        help="Total de interacciones (steps) a entrenar")
    args = parser.parse_args()

    train(resume_path=args.resume, total_steps=args.steps)
