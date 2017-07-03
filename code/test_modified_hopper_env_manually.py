import gym
from test_manual import test

arg_dict = dict(id="HopperHalfGravity-v0",
                entry_point="modified_gravity_hopper:GravityEnv",
                max_episode_steps=1000,
                kwargs={"gravity" : -1.0})

gym.envs.register(**arg_dict)

test("HopperHalfGravity-v0")
