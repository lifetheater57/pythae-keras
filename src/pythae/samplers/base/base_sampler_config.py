from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseSamplerConfig(BaseConfig):
    """
    BaseSampler config class.
    """

    pass
