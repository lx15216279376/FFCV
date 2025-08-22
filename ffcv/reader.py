import numpy as np

from .utils import decode_null_terminated_string
from .types import (ALLOC_TABLE_TYPE, HeaderType, CURRENT_VERSION,
                    FieldDescType, get_handlers, get_metadata_type)

# 二进制数据文件读取器
class Reader:

    def __init__(self, fname, custom_handlers={}):
        self._fname = fname
        self._custom_handlers = custom_handlers
        self.read_header()  # 读取文件头信息
        self.read_field_descriptors()   # 读取字段描述符
        self.read_metadata()    # 读取元数据
        self.read_allocation_table()    # 读取分配表

    @property
    def file_name(self):
        return self._fname

    def read_header(self):
        # 从文件中读取头部信息
        # 头部信息包含版本号、样本数量、页面大小和字段数量等
        header = np.fromfile(self._fname, dtype=HeaderType, count=1)[0]
        header.setflags(write=False)
        version = header['version']

        if version != CURRENT_VERSION:
            msg = f"file format mismatch: code={CURRENT_VERSION},file={version}"
            raise AssertionError(msg)

        self.num_samples = header['num_samples']    # 样本数量
        self.page_size = header['page_size']    # 页面大小
        self.num_fields = header['num_fields']  # 字段数量
        self.header = header    # 保存头部信息

    # 读取字段描述信息并初始化字段处理器
    def read_field_descriptors(self):
        offset = HeaderType.itemsize
        # 从文件中读取字段描述符
        field_descriptors = np.fromfile(self._fname, dtype=FieldDescType,
                                        count=self.num_fields, offset=offset)
        field_descriptors.setflags(write=False)
        # 获取字段处理器
        handlers = get_handlers(field_descriptors)

        # 初始化字段描述符和处理器
        self.field_descriptors = field_descriptors
        self.field_names = list(map(decode_null_terminated_string,
                                    self.field_descriptors['name']))    # 解码字段名
        self.handlers = dict(zip(self.field_names, handlers))

        # 处理自定义字段
        for field_name, field_desc in zip(self.field_names, self.field_descriptors):
            if field_name in self._custom_handlers:
                CustomHandler = self._custom_handlers[field_name]
                self.handlers[field_name] = CustomHandler.from_binary(field_desc['arguments'])
        
        for field_name, handler in self.handlers.items():
            if handler is None:
                raise ValueError(f"Must specify a custom_field entry " \
                                 f"for custom field {field_name}")

        # 根据字段处理器获取元数据类型
        self.metadata_type = get_metadata_type(list(self.handlers.values()))

    # 读取元数据
    def read_metadata(self):
        offset = HeaderType.itemsize + self.field_descriptors.nbytes
        self.metadata = np.fromfile(self._fname, dtype=self.metadata_type,
                                   count=self.num_samples, offset=offset)
        self.metadata.setflags(write=False)

    # 读取分配表
    # 分配表用于记录每个样本在文件中的偏移位置
    # 通过分配表可以快速定位样本数据
    # 分配表的每一项包含样本索引和对应的偏移量
    # 分配表的偏移量存储在头部信息中
    def read_allocation_table(self):
        offset = self.header['alloc_table_ptr']
        alloc_table = np.fromfile(self._fname, dtype=ALLOC_TABLE_TYPE,
                                  offset=offset)
        alloc_table.setflags(write=False)
        self.alloc_table = alloc_table


