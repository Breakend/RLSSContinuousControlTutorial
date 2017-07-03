import argparse
import os.path as osp
import pickle

import tensorflow as tf

from ddpg.ddpg import DDPG
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc import ext
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.deterministic_mlp_policy import \
    DeterministicMLPPolicy
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import \
    ContinuousMLPQFunction

parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--data_dir", default="./data/")
parser.add_argument("--reward_scale", default=1.0, type=float)
parser.add_argument("--use_ec2", action="store_true", help="Use your ec2 instances if configured")
parser.add_argument("--dont_terminate_machine", action="store_false", help="Whether to terminate your spot instance or not. Be careful.")
args = parser.parse_args()

stub(globals())
ext.set_seed(1)

gymenv = GymEnv(args.env, force_reset=True, record_video=True, record_log=True)

env = TfEnv(normalize(gymenv))

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    name="policy",
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)

es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_sizes=(100,100),
                            hidden_nonlinearity=tf.nn.relu,)

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=64,
    max_path_length=env.horizon,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=args.num_epochs,
    discount=0.99,
    scale_reward=args.reward_scale,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
    plot=False
)


run_experiment_lite(
    algo.train(),
    log_dir=None if args.use_ec2 else args.data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="DDPG_" + args.env,
    seed=1,
    mode="ec2" if args.use_ec2 else "local",
    plot=False,
    # dry=True,
    terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)
