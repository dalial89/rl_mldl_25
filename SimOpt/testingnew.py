import time

MAX_STEPS = 2000  # reasonable cap for Hopper

for ep in range(episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    start_time = time.time()

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # safety check
        if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
            print(f"[WARNING] Episode {ep+1}: Non-finite reward or obs detected.")
            break

        total_reward += reward
        steps += 1

        # emergency timeout if step loop takes too long
        if time.time() - start_time > 10:  # seconds
            print(f"[WARNING] Episode {ep+1} taking too long... breaking.")
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    print(f"[DEBUG] Episode {ep+1} — Reward: {total_reward:.2f} | Steps: {steps}")
