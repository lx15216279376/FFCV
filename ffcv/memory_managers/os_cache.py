from typing import TYPE_CHECKING

import numpy as np
import numba as nb

from .base import MemoryManager, MemoryContext
from ..pipeline.compiler import Compiler

if TYPE_CHECKING:
    from ..reader import Reader

# =================================================================
# 1. OSCacheContext：负责“一个 epoch” 的生命周期
#    - 只做一件事：把 .beton 文件一次性 mmap 成只读 uint8 数组
#    - 由于 mmap 不会写文件，所以 __exit__ 里什么都不用做
# =================================================================
class OSCacheContext(MemoryContext):
    def __init__(self, manager:MemoryManager):
        self.manager = manager
        self.mmap = None

    @property
    def state(self):
        return (self.mmap, self.manager.ptrs, self.manager.sizes)

    def __enter__(self):
        res = super().__enter__()
        if self.mmap is None:
            self.mmap = np.memmap(self.manager.reader.file_name,
                                  'uint8', mode='r')
        return res

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Numpy doesn't have an API to close memory maps yet
        # The only thing one can do is flush it be since we are not
        # Writing to it it's pointless
        # Moreover we want to avoid opening the memmap over and over
        # anyway.
        return super().__exit__(__exc_type, __exc_value, __traceback)

# =================================================================
# 2. OSCacheManager：实现 MemoryManager 接口
#    - schedule_epoch() 永远返回同一个 OSCacheContext 实例
#    - 负责告诉 compiler “pipeline 里读盘函数的类型签名”
# =================================================================
class OSCacheManager(MemoryManager):

    def __init__(self, reader: 'Reader'):
        super().__init__(reader)
        self.context = OSCacheContext(self)

    def schedule_epoch(self, schedule):
        return self.context

    @property
    def state_type(self):
        t1 = nb.uint8[::1]
        t1.multable = False
        t2 = nb.uint64[::1]
        t1.mutable = False
        return nb.types.Tuple([t1, t2, t2])

    def compile_reader(self):
        def read(address, mem_state):
            size = mem_state[2][np.searchsorted(mem_state[1], address)]
            return mem_state[0][address:address + size]

        return Compiler.compile(read, nb.uint8[::1](nb.uint64, self.state_type))

