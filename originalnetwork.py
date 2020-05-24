import gym
from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


import numpy as np
import os
import itertools

import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.tf_util import get_session
from baselines.common import set_global_seeds


from monitoring_rewards.core import TraceStep
from monitoring_rewards.multi_reward_monitor import MultiRewardMonitor
from monitoring_rewards.monitoring_specification import MonitoringSpecification
import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
import tensorflow.contrib.layers as layers



from gym_breakout_pygame.wrappers.normal_space import BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")
        path="./model.pkl"

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.
    Parameters
    ----------
    path: str
        path to the act function pickle
    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=3000,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=3000,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):


    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)


    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        # gamma=gamma,
        # grad_norm_clipping=10,
        # param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(100000),
                                 initial_p=1.0,
                                 final_p=0.02)

    # Initialize the paramete    print(type(act))rs and copy them to the target network.
    U.initialize()
    update_target()





    old_state = None





    formula_LTLf_1 = "!F(die)"
    monitoring_RightToLeft = MonitoringSpecification(
        ltlf_formula=formula_LTLf_1,
        r=1,
        c=-10,
        s=1,
        f=-10
    )



    monitoring_specifications = [monitoring_RightToLeft]

    stepCounter = 0
    done = False

    def RightToLeftConversion(observation) -> TraceStep:

        print(stepCounter)


        if(done and not(stepCounter>=199)):
            die=True
        else:
            die=False


        dictionary={'die': die}
        print(dictionary)
        return dictionary

    multi_monitor = MultiRewardMonitor(
        monitoring_specifications=monitoring_specifications,
        obs_to_trace_step=RightToLeftConversion
    )













    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True


    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        episodeCounter=0
        num_episodes=0

        for t in itertools.count():
            
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            #print(action)
            new_obs, rew, done, _ = env.step(action)
            stepCounter+=1

            rew, is_perm = multi_monitor(new_obs)
            old_state=new_obs




            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew


            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if episodeCounter % 100 == 0 or episodeCounter<1:
                # Show off the result
                #print("coming here Again and Again")
                env.render()


            if done:
                episodeCounter+=1
                num_episodes+=1
                obs = env.reset()
                episode_rewards.append(0)
                multi_monitor.reset()
                stepCounter=0
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean 100 episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 500 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    act.save_act()
                    #save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        # if model_saved:
        #     if print_freq is not None:
        #         logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
        #     load_variables(model_file)

    return act


def model(inpt,  reuse=False):
    """This model takes as input an observation and returns values of all actions."""

    out = inpt
    out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)

    out = layers.fully_connected(out, num_outputs=2, activation_fn=None)
    return out

env = gym.make("CartPole-v0")
def main():
    print('-*-*-*- train worker -*-*-*-')

    
    print(env.action_space.sample())
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space.n)

    #env=BreakoutNMultiDiscrete()
    #model = deepq.models.mlp([64])



    act = learn(
        env,
        network=model,
        lr=5e-4,
        max_timesteps=100000,
        checkpoint_freq=1000,
        buffer_size=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        load_state="cartpole_model.pkl"
    )

    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
