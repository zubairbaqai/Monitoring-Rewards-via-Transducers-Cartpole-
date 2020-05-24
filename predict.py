import gym
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import deepq
from gym_breakout_pygame.wrappers.normal_space import BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace







def main():
    print('-*-*-*- enjoy worker -*-*-*-')
    # tf.graph().as_default()
    # tf.reset_default_graph()
    env = gym.make("CartPole-v0")
    #env=BreakoutNMultiDiscrete()
    act = deepq.load_act("model.pkl")

    max_episodes = 5

    while max_episodes > 0:
        obs, done = env.reset(), False
        #print(obs)
        episode_rew = 0
        counter=0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            counter=counter+1
        print(counter)
        print("Episode reward", episode_rew)
        max_episodes = max_episodes - 1


if __name__ == '__main__':
    main()