from pyjava.udf import UDFMaster,UDFWorker

import time
import asyncio
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import pytest


def init_func(model_refs: List[ClientObjectRef],conf: Dict[str, str],):
    pass

def apply_func(model,v):
    return 0

class TestUDFMaster(object):
    conf = {"UDF_CLIENT":"chat"}

    def setup_class(self):
        if ray.is_initialized():
            ray.shutdown()        
        ray.init(address="auto",namespace="default")        
        conf = self.conf
        udf_name = conf["UDF_CLIENT"]
        max_concurrency = int(conf.get("maxConcurrency", "2"))
        masterMaxConcurrency = int(conf.get("masterMaxConcurrency", "1000"))        

        try:
            stop_flag = True
            counter = 30
            temp_udf_master = ray.get_actor(udf_name)
            ray.kill(temp_udf_master)
            while stop_flag or counter < 0:
                time.sleep(1)
                try:
                    ray.get_actor(udf_name)
                except Exception:
                    stop_flag = False
                    counter = counter - 1

        except Exception as inst:
            print(inst)
            pass
                
        UDFMaster.options(name=udf_name, lifetime="detached",
                          max_concurrency=masterMaxConcurrency).remote(
            max_concurrency, conf, init_func, apply_func)
        
        temp_udf_master = ray.get_actor(udf_name)
        # init workers
        ray.get(temp_udf_master.create_workers.remote(conf))
        
        # build model in every worker
        build_model_jobs = [worker.build_model.remote() for worker in ray.get(temp_udf_master.workers.remote())]
        ray.get(build_model_jobs)                
                                       
    def teardown_class(self):
        ray.kill(ray.get_actor(self.conf["UDF_CLIENT"]))
        if ray.is_initialized():
            ray.shutdown()  

    @pytest.mark.asyncio
    async def test_schedule(self):
        conf = {"UDF_CLIENT":"chat"}
        udf_name = conf["UDF_CLIENT"]
         
        # get worker and input value are all io operation,
        # so we use asyncio/threading to fetch them in parallel and not block the main thread.        
        def get_worker(udf_name):
            udf_master = ray.get_actor(udf_name)
            [index, worker] = ray.get(udf_master.get.remote())
            is_coroutine = ray.get(worker.is_coroutine.remote())
            return udf_master,index,worker,is_coroutine
        def get_input():
            input_value = [row["value"] for row in [{"value":{}}]]
            return input_value
        
               
        task1 = asyncio.to_thread(get_input)
        task2 = asyncio.to_thread(get_worker,udf_name)        

        results = await asyncio.gather(task1, task2)
        input_value = results[0]
        udf_master,index,worker,is_coroutine = results[1]
        print("worker ",index) 
        assert(index == 0)
        ray.get(udf_master.give_back.remote(index))


        task2 = asyncio.to_thread(get_worker,udf_name)
        results = await asyncio.gather(task2)
        udf_master,index,worker,is_coroutine = results[0]
        ray.get(udf_master.give_back.remote(index))
        assert(index == 1)

        task2 = asyncio.to_thread(get_worker,udf_name)
        results = await asyncio.gather(task2)
        udf_master,index,worker,is_coroutine = results[0]
        ray.get(udf_master.give_back.remote(index))
        assert(index == 0)
        
        


                    
        
        
        
        

       
    



    
