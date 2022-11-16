import itertools
from typing import Any, Generic, List, Callable, Union, Tuple, Iterable

import ray
import pyarrow as pa
from ray.data.datasource.datasource import WriteResult
from ray.types import ObjectRef

from pyjava.api.mlsql import RayContext
from ray.data import Datasource, ReadTask
from ray.data.block import Block, BlockMetadata
from ray.data.impl.arrow_block import ArrowRow


class KoloRawDatasource(object):
    def __init__(self, data_refs: List[str], num_tables_per_block: int = 1):
        self.data_refs = data_refs
        self.block_refs = []

        def merged_pa_generator(pa_generator):
            merged_tables = []
            while True:
                try:
                    for i in range(0, 7):
                        patable = next(pa_generator)
                        merged_tables.append(patable)
                    yield pa.concat_tables(merged_tables)
                    merged_tables.clear()
                except StopIteration as e:
                    if len(merged_tables) > 0:
                        yield pa.concat_tables(merged_tables)
                    print("Reading from pa table iterator is done!")
                    break

        @ray.remote
        def make_block(_data_ref):
            block_refs = []
            data_iter = RayContext.fetch_data_from_single_data_server_as_arrow(_data_ref)
            temp_box = []

            merged_data_iters = merged_pa_generator(data_iter)

            for arrow_table in merged_data_iters:
                temp_box.append(arrow_table)
                if len(temp_box) == num_tables_per_block:
                    for t in temp_box:
                        block_refs.append(ray.put(t))
                    temp_box.clear()

            if len(temp_box) != 0:
                for t in temp_box:
                    block_refs.append(ray.put(t))
                temp_box.clear()
            return block_refs

        for data_ref in data_refs:
            for item in ray.get(make_block.remote(data_ref)):
                self.block_refs.append(item)

    def to_dataset(self):
        return ray.data.from_arrow_refs(self.block_refs)


class KoloDatasource(Datasource[Union[ArrowRow]]):

    def do_write(self, blocks: List[ObjectRef[Block]],
                 metadata: List[BlockMetadata], **write_args) -> List[ObjectRef[WriteResult]]:
        raise NotImplementedError

    def prepare_read(self, parallelism: int,
                     data_refs: List[str], meta: Union[type, "pyarrow.lib.Schema"]) -> List[ReadTask]:
        read_tasks: List[ReadTask] = []
        for data_ref in data_refs:
            def make_block(_data_ref):
                _data_iter = RayContext.fetch_data_from_single_data_server_as_arrow(_data_ref)
                return _data_iter

            meta = BlockMetadata(
                num_rows=None,
                size_bytes=None,
                schema=meta,
                input_files=None)

            read_tasks.append(
                ReadTask(
                    lambda _data_ref=data_ref: make_block(_data_ref), meta))

        return read_tasks
