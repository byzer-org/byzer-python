from typing import Any, Generic, List, Callable, Union, Tuple, Iterable

import ray
from pyjava.api.mlsql import RayContext


class KoloRawDatasource(object):
    def __init__(self, data_refs: List[str], num_tables_per_block: int = 1):
        self.data_refs = data_refs
        self.block_refs = []

        @ray.remote
        def make_block(_data_ref):
            block_refs = []
            data_iter = RayContext.fetch_data_from_single_data_server_as_arrow(_data_ref)
            temp_box = []
            for arrow_table in data_iter:
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
