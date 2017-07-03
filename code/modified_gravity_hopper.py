import math
import os
import os.path as osp
import random
import tempfile
import xml.etree.ElementTree as ET

import gym
import mujoco_py
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.hopper import HopperEnv


class GravityEnv(HopperEnv, utils.EzPickle):
    """
    Allows the gravity to be changed by the
    """
    def __init__(
            self,
            gravity=-9.81,
            *args,
            **kwargs):
        HopperEnv.__init__(self)
        utils.EzPickle.__init__(self)

        # make sure we're using a proper OpenAI gym Mujoco Env
        assert isinstance(self, mujoco_env.MujocoEnv)

        self.model.opt.gravity = (mujoco_py.mjtypes.c_double * 3)(*[0., 0., gravity])
        self.model._compute_subtree()
        self.model.forward()
