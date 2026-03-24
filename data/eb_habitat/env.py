# data/eb_habitat/env.py
from data.base_env import StaticEnv

class EBHabitatEnv(StaticEnv):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def compute_reward(cls, completions: list[str], solution: list[str], **kwargs) -> list[float]:
        # SFT 阶段不使用 Reward，返回 0.0
        return [0.0 for _ in completions]