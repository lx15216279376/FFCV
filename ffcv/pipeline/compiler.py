import pdb
from numba import njit, set_num_threads, prange, warnings as nwarnings, get_num_threads
from numba.core.errors import NumbaPerformanceWarning
from multiprocessing import cpu_count
import torch as ch
import warnings
from os import sched_getaffinity

# 编译工具类，用于管理Numba和Pytorch的编译选项和线程设置
class Compiler:

    # 类变量，用于控制编译器是否启用
    @classmethod
    def set_enabled(cls, b):
        cls.is_enabled = b

    # 设置Numba和Pytorch的线程数
    # 如果n小于1，则使用CPU核心数
    @classmethod
    def set_num_threads(cls, n):
        if n < 1 :
            n = len(sched_getaffinity(0))
        cls.num_threads = n
        set_num_threads(n)
        ch.set_num_threads(n)

    # 编译给定的函数代码
    @classmethod
    def compile(cls, code, signature=None):
        parallel = False
        if hasattr(code, 'is_parallel'):
            parallel = code.is_parallel and cls.num_threads > 1
        
        if cls.is_enabled:
            return njit(signature, # 函数签名
                        fastmath=True, # 启用快速数学运算
                        nogil=True, # 释放GIL锁，允许并行
                        error_model='numpy', # 使用Numpy的错误处理模型
                        parallel=parallel # 是否并行执行
            )(code)
        return code

    # 获取适合当前线程数的迭代器
    @classmethod
    def get_iterator(cls):
        if cls.num_threads > 1:
            return prange
        else:
            return range

Compiler.set_enabled(True)  # 启用编译器
Compiler.set_num_threads(1) # 设置线程数为1（可以根据需要调整）
