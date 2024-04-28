import unittest
from pyjava.udf.udf_master import UDFMaster
from pyjava.udf.udf_worker import UDFWorker
import ray
import pytest

@pytest.fixture
def init_ray():
    # Start Ray
    ray.init(ignore_reinit_error=True)
    yield
    # Shutdown Ray
    ray.shutdown()

def init_func(model_refs, conf):
    return "model"

def apply_func(model, data):
    return data
        
def test_reload_worker(init_ray):
    udf_master = UDFMaster.remote(3, {}, init_func, apply_func)
    ray.get(udf_master.create_workers.remote({}))
    
    old_worker = ray.get(udf_master.actors[0])
    ray.get(udf_master.reload_worker.remote(0))
    new_worker = ray.get(udf_master.actors[0])
    
    assert old_worker != new_worker
    
    # Test exception when modelServers in conf
    udf_master = UDFMaster.remote(3, {"modelServers":"ms"}, init_func, apply_func)
    ray.get(udf_master.create_workers.remote({"modelServers":"ms"}))
    with pytest.raises(Exception):
        ray.get(udf_master.reload_worker.remote(0))