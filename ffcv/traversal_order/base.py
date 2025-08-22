from abc import ABC, abstractmethod
from typing import Sequence
from ..reader import Reader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..loader.main_thread import Loader

# 数据遍历顺序的抽象基类（定义如何遍历数据集）
class TraversalOrder(ABC):
    def __init__(self, loader: 'Loader'):
        self.loader = loader
        self.indices = self.loader.indices
        self.seed = self.loader.seed
        self.distributed = loader.distributed
        self.sampler = None

    @abstractmethod
    def sample_order(self, epoch:int) -> Sequence[int]:
        raise NotImplemented()
