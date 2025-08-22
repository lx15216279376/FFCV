import torch as ch

from typing import List, Union
from .operation import Operation
from ..transforms.module import ModuleWrapper
from ..transforms import ToTensor

# 管道规范类，用于定义数据处理管道的规范
class PipelineSpec:
    # 初始化管道规范
    # source: 数据源，可以是字符串路径或操作对象
    # decoder: 解码器操作（可选）
    # transforms: 其他转换操作列表（可选）
    def __init__(self, source: Union[str, Operation], decoder: Operation = None,
                 transforms:List[Operation] = None ):

        self.source = source
        self.decoder = decoder
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.default_pipeline = (decoder is None
                                 and not transforms
                                 and isinstance(source, str))

    def __repr__(self):
        return repr((self.source, self.decoder, self.transforms))

    def __str__(self):
        return self.__repr__()

    # 接受并整合解码器到管道中
    def accept_decoder(self, Decoder, output_name):
        if not isinstance(self.source, str) and self.decoder is not None:
            raise ValueError("Source can't be a node and also have a decoder")

        if Decoder is not None:
            # The first element of the operations is a decoder
            # 情况1：转换列表的第一个操作已经是所需解码器，则提取它
            if self.transforms and isinstance(self.transforms[0], Decoder):
                self.decoder = self.transforms.pop(0)
            # 情况2：转换列表为空或第一个操作不是所需解码器，则创建新的解码器实例
            elif self.decoder is None:
                try:
                    self.decoder = Decoder()
                except Exception:
                    msg = f"Impossible to use default decoder for {output_name},"
                    msg += "make sure you specify one in your pipeline."
                    raise ValueError(msg)

        if self.default_pipeline:
            self.transforms.append(ToTensor())

        for i, op in enumerate(self.transforms):
            if isinstance(op, ch.nn.Module):
                self.transforms[i] = ModuleWrapper(op)
            