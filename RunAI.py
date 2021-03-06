
import numpy as np
import matplotlib.pyplot as plt
import gym
from RL_PPO import RL


EP_MAX = 10
EP_LEN = 200
GAMMA = 0.9
BATCH = 32
GYM_NAME = 'Pendulum-v0'

env = gym.make(GYM_NAME).unwrapped
RL_Network = RL()
all_episode_reward = []

for ep in range(EP_MAX):
    state = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    episode_reward = 0
    for t in range(EP_LEN):    # in one episode
        #env.render()
        action = RL_Network.choose_action(state)
        state_new, reward, done, _ = env.step(action)
        buffer_s.append(state)
        buffer_a.append(action)
        buffer_r.append((reward+8)/8)    # normalize reward, find to be useful
        state = state_new
        episode_reward += reward

        # update RL_Network
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            value_state_new = RL_Network.get_value(state_new)
            discounted_reward = []
            for reward in buffer_r[::-1]:
                value_state_new = reward + GAMMA * value_state_new
                discounted_reward.append(value_state_new)
            discounted_reward.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_reward)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            RL_Network.update(bs, ba, br)
    if ep == 0: all_episode_reward.append(episode_reward)
    else: all_episode_reward.append(all_episode_reward[-1]*0.9 + episode_reward*0.1)
    print(
        'Episode: %i' % ep,
        "|episode_reward: %i" % episode_reward,
        #("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )
env.close()
plt.plot(np.arange(len(all_episode_reward)), all_episode_reward)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()