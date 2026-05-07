import sys
sys.path.insert(0,".")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from rl.env import RobotReachEnv

def main():
    print("Checking environment..")
    env = RobotReachEnv()
    check_env(env)
    print(("Environment Ok\n"))

    eval_env = RobotReachEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./logs/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    print("Starting training..")
    model.learn(total_timesteps=500_000,callback=eval_callback)

    model.save("model/ppo_reach")
    print("\nTraining complete.")

    env.close()
    eval_env.close()

if __name__ =="__main__":
    main()