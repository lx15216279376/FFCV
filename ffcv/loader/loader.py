"""
FFCV loader
"""
import enum
from os import environ, sched_getaffinity
import ast
from multiprocessing import cpu_count
from re import sub
from typing import Any, Callable, Mapping, Sequence, Type, Union, Literal
from collections import defaultdict
from collections.abc import Collection
from enum import Enum, unique, auto

from ffcv.fields.base import Field

import torch as ch
import numpy as np

from .epoch_iterator import EpochIterator
from ..reader import Reader
from ..traversal_order.base import TraversalOrder
from ..traversal_order import Random, Sequential, QuasiRandom
from ..pipeline import Pipeline, PipelineSpec, Compiler
from ..pipeline.operation import Operation
from ..pipeline.graph import Graph
from ..memory_managers import (
    ProcessCacheManager, OSCacheManager, MemoryManager
)

# 定义遍历顺序选项的枚举类型
@unique
class OrderOption(Enum):
    SEQUENTIAL = auto() # 顺序遍历
    RANDOM = auto() # 随机遍历
    QUASI_RANDOM = auto() # 准随机遍历

# 定义遍历顺序类型的别名
ORDER_TYPE = Union[
    TraversalOrder,
    Literal[OrderOption.SEQUENTIAL,
            OrderOption.RANDOM]

]

# 定义遍历顺序映射，将OrderOption映射到具体的TraversalOrder实现
ORDER_MAP: Mapping[ORDER_TYPE, TraversalOrder] = {
    OrderOption.RANDOM: Random,
    OrderOption.SEQUENTIAL: Sequential,
    OrderOption.QUASI_RANDOM: QuasiRandom
}

# 默认的进程缓存和操作系统缓存设置
DEFAULT_PROCESS_CACHE = int(environ.get('FFCV_DEFAULT_CACHE_PROCESS', "0"))
DEFAULT_OS_CACHE = not DEFAULT_PROCESS_CACHE

class Loader:
    """FFCV loader class that can be used as a drop-in replacement
    for standard (e.g. PyTorch) data loaders.

    Parameters
    ----------
    fname: str
        Full path to the location of the dataset (.beton file format).
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers used for data loading. Consider using the actual number of cores instead of the number of threads if you only use JITed augmentations as they usually don't benefit from hyper-threading.
    os_cache : bool
        Leverages the operating for caching purposes. This is beneficial when there is enough memory to cache the dataset and/or when multiple processes on the same machine training using the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    order : Union[OrderOption, TraversalOrder]
        Traversal order, one of: SEQUENTIAL, RANDOM, QUASI_RANDOM, or a custom TraversalOrder

        QUASI_RANDOM is a random order that tries to be as uniform as possible while minimizing the amount of data read from the disk. Note that it is mostly useful when `os_cache=False`. Currently unavailable in distributed mode.
    distributed : bool
        For distributed training (multiple GPUs). Emulates the behavior of DistributedSampler from PyTorch.
    seed : int
        Random seed for batch ordering.
    indices : Sequence[int]
        Subset of dataset by filtering only some indices.
    pipelines : Mapping[str, Sequence[Union[Operation, torch.nn.Module]]
        Dictionary defining for each field the sequence of Decoders and transforms to apply.
        Fileds with missing entries will use the default pipeline, which consists of the default decoder and `ToTensor()`,
        but a field can also be disabled by explicitly by passing `None` as its pipeline.
    custom_fields : Mapping[str, Field]
        Dictonary informing the loader of the types associated to fields that are using a custom type.
    drop_last : bool
        Drop non-full batch in each iteration.
    batches_ahead : int
        Number of batches prepared in advance; balances latency and memory.
    recompile : bool
        Recompile every iteration. This is necessary if the implementation of some augmentations are expected to change during training.
    """
    """FFCV数据加载器，可作为标准数据加载器(如PyTorch)的替代品

    参数说明
    ----------
    fname: str
        数据集文件路径(.beton格式)
    batch_size : int
        批大小
    num_workers : int
        数据加载工作进程数。如果只使用JIT编译的增强操作，建议设置为实际核心数而非线程数
    os_cache : bool
        是否使用操作系统缓存。当有足够内存缓存整个数据集和/或多进程使用相同数据集时建议启用
    order : Union[OrderOption, TraversalOrder]
        数据遍历顺序，可选: SEQUENTIAL(顺序), RANDOM(随机), QUASI_RANDOM(准随机)或自定义TraversalOrder
    distributed : bool
        是否分布式训练(多GPU)。模拟PyTorch的DistributedSampler行为
    seed : int
        随机排序的种子
    indices : Sequence[int]
        数据集子集的索引列表
    pipelines : Mapping[str, Sequence[Union[Operation, torch.nn.Module]]
        字段处理管道定义，指定每个字段的解码器和转换操作序列
    custom_fields : Mapping[str, Field]
        自定义字段类型映射
    drop_last : bool
        是否丢弃最后不完整的批次
    batches_ahead : int
        预准备的批次数，平衡延迟和内存使用
    recompile : bool
        是否每次迭代重新编译(当增强操作实现可能变化时需要)
    """
    def __init__(self,
                 fname: str,
                 batch_size: int,
                 num_workers: int = -1,
                 os_cache: bool = DEFAULT_OS_CACHE,
                 order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = None,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 pipelines: Mapping[str,
                                    Sequence[Union[Operation, ch.nn.Module]]] = {},
                 custom_fields: Mapping[str, Type[Field]] = {},
                 drop_last: bool = True,
                 batches_ahead: int = 3,
                 recompile: bool = False,  # Recompile at every epoch
                 order_kwargs: dict = dict(),
                 ):

        # 分布式训练随机种子处理
        if distributed and order == OrderOption.RANDOM and (seed is None):
            print('Warning: no ordering seed was specified with distributed=True. '
                  'Setting seed to 0 to match PyTorch distributed sampler.')
            seed = 0
        elif seed is None:
            tinfo = np.iinfo('int32')
            seed = np.random.randint(0, tinfo.max)

        # We store the original user arguments to be able to pass it to the
        # filtered version of the datasets
        # 存储原始用户参数，以便在过滤数据集时使用
        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'os_cache': os_cache,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile
        }
        # 初始化基本属性
        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname, custom_fields)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code = None
        self.recompile = recompile

        # 设置工作线程数
        if self.num_workers < 1:
            self.num_workers = len(sched_getaffinity(0))

        Compiler.set_num_threads(self.num_workers)

        # 处理数据集索引
        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        # 初始化内存管理器
        if os_cache:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        else:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)

        # 初始化遍历顺序策略
        if order in ORDER_MAP:
            self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        elif issubclass(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self, **order_kwargs)
        else:
            raise ValueError(f"Order {order} is not a supported order type or a subclass of TraversalOrder")

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch: int = 0

        # 初始化管道相关属性
        self.pipelines = {}
        self.pipeline_specs = {}
        self.field_name_to_f_ix = {}   # 字段名到字段索引的映射
        
        custom_pipeline_specs = {}  # 用户自定义管道

        # Creating PipelineSpec objects from the pipeline dict passed
        # by the user
        for output_name, spec in pipelines.items():
            if isinstance(spec, PipelineSpec):
                pass
            elif isinstance(spec, Sequence):
                spec = PipelineSpec(output_name, decoder=None, transforms=spec)
            elif spec is None:
                continue  # This is a disabled field
            else:
                msg = f"The pipeline for {output_name} has to be "
                msg += f"either a PipelineSpec or a sequence of operations"
                raise ValueError(msg)
            custom_pipeline_specs[output_name] = spec

        # Adding the default pipelines
        # 添加默认数据处理管道
        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            self.field_name_to_f_ix[field_name] = f_ix

            if field_name not in custom_pipeline_specs:
                # We add the default pipeline
                if field_name not in pipelines:
                    self.pipeline_specs[field_name] = PipelineSpec(field_name)
            else:
                self.pipeline_specs[field_name] = custom_pipeline_specs[field_name]

        # We add the custom fields after the default ones
        # This is to preserve backwards compatibility and make sure the order
        # is intuitive
        # 添加自定义字段管道
        for field_name, spec in custom_pipeline_specs.items():
            if field_name not in self.pipeline_specs:
                self.pipeline_specs[field_name] = spec

        # 构建处理管道图
        self.graph = Graph(self.pipeline_specs, self.reader.handlers,
                           self.field_name_to_f_ix, self.reader.metadata,
                           memory_read)
        
        # 生成处理代码
        self.generate_code()
        self.first_traversal_order = self.next_traversal_order()    # 初始遍历顺序

    # 生成下一轮的遍历顺序
    def next_traversal_order(self):
        return self.traversal_order.sample_order(self.next_epoch)

    # 实现迭代器接口
    def __iter__(self):
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        selected_order = order[:len(self) * self.batch_size]
        self.next_epoch += 1

        # Compile at the first epoch
        if self.code is None or self.recompile:
            self.generate_code()

        return EpochIterator(self, selected_order)

    # 过滤数据集
    def filter(self, field_name:str, condition: Callable[[Any], bool]) -> 'Loader':
        new_args = {**self._args}
        pipelines = {}

        # Disabling all the other fields
        for other_field_name in self.reader.handlers.keys():
            pipelines[other_field_name] = None

        # We reuse the original pipeline for the field we care about
        try:
            pipelines[field_name] = new_args['pipelines'][field_name]
        except KeyError:
            # We keep the default one if the user didn't setup a custom one
            del pipelines[field_name]
            pass

        new_args['pipelines'] = pipelines

        # We use sequential order for speed and to know which index we are
        # filtering
        new_args['order'] = OrderOption.SEQUENTIAL
        new_args['drop_last'] = False
        sub_loader = Loader(**new_args)
        selected_indices = []

        # Iterate through the loader and test the user defined condition
        for i, (batch,) in enumerate(sub_loader):
            for j, sample in enumerate(batch):
                sample_id = i * self.batch_size + j
                if condition(sample):
                    selected_indices.append(sample_id)

        final_args = {**self._args}
        final_args['indices'] = np.array(selected_indices)
        return Loader(**final_args)


    def __len__(self):
        next_order = self.first_traversal_order
        if self.drop_last:
            return len(next_order) // self.batch_size
        else:
            return int(np.ceil(len(next_order) / self.batch_size))


    # 生成处理代码
    def generate_code(self):
        queries, code = self.graph.collect_requirements()
        self.code = self.graph.codegen_all(code)
        

