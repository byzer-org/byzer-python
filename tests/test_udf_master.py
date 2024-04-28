import pytest
from pyjava.udf.udf_master import UDFMaster
from pyjava.udf.udf_worker import UDFWorker

def init_func(model_refs, conf):
    return "Test Model"

def apply_func(model, data):
    return f"Result: {data}"

@pytest.mark.asyncio
async def test_create_worker():
    conf = {
        "num_cpus": "1",
        "infer_backend": "transformers",
        "UDF_CLIENT": "Test Model"
    }
    
    udf_master = UDFMaster.remote(1, conf, init_func, apply_func)
    worker = await udf_master.create_worker.remote(0, conf)
    
    assert isinstance(worker, UDFWorker)
    
    await worker.build_model.remote()
    result = await worker.apply.remote("Test Data")
    
    assert result == "Result: Test Data"
