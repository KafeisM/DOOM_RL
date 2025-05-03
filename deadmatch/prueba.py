from DoomGame import DoomEnv
from env_wrappers import RewardShapingWrapper, PreprocessFrame, FrameStack

# 1) Entorno base
env0 = DoomEnv(config_file="deathmatch.cfg", frame_skip=4)
obs0, _ = env0.reset()
print("❶ DoomEnv.reset()    →", type(obs0),
      " keys:", getattr(obs0, "keys", lambda: [])(),
      " screen shape:", obs0['screen'].shape)

# 2) Reward shaping
env1 = RewardShapingWrapper(env0)
obs1, _ = env1.reset()
print("❷ RewardShapingWrapper.reset() →", type(obs1),
      " keys:", getattr(obs1, "keys", lambda: [])(),
      " screen shape:", obs1['screen'].shape)

# 3) Preprocesado
env2 = PreprocessFrame(env1, width=108, height=60, grayscale=False)
obs2, _ = env2.reset()
print("❸ PreprocessFrame.reset() →", type(obs2),
      obs2.shape)

# 4) Frame stacking
env3 = FrameStack(env2, n_frames=4)
obs3, _ = env3.reset()
print("❹ FrameStack.reset() →", type(obs3),
      obs3.shape)
