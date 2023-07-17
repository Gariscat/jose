import gym
from gym.spaces import Space
from typing import Optional, Union
import mujoco


DEFAULT_SIZE = 480


class SinglePlayerEnv(gym.Env):
    def __init__(
        self,
        model_path: str,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ) -> None:
        self.width = width
        self.height = height
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)