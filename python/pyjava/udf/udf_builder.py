
import uuid
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientObjectRef
from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar
from pyjava.udf.udf_master import UDFMaster
import asyncio

class UDFBuilder(object):
    @staticmethod
    def build(ray_context: RayContext,
              init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
              apply_func: Callable[[Any, Any], Any]) -> NoReturn:
        conf = ray_context.conf()
        udf_name = conf["UDF_CLIENT"]
        max_concurrency = int(conf.get("maxConcurrency", "3"))
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
        ray_context.build_result([])

    @staticmethod
    async def _async_apply(ray_context: RayContext):
        conf = ray_context.conf()
        udf_name = conf["UDF_CLIENT"]
         
        # get worker and input value are all io operation,
        # so we use asyncio/threading to fetch them in parallel and not block the main thread.        
        def get_worker(udf_name):
            udf_master = ray.get_actor(udf_name)
            [index, worker] = ray.get(udf_master.get.remote())
            is_coroutine = ray.get(worker.is_coroutine.remote())
            return udf_master,index,worker,is_coroutine
        def get_input(ray_context):
            input_value = [row["value"] for row in ray_context.python_context.fetch_once_as_rows()]
            return input_value
        
               
        task1 = asyncio.to_thread(get_input,ray_context)
        task2 = asyncio.to_thread(get_worker,udf_name)        

        results = await asyncio.gather(task1, task2)
        input_value = results[0]
        udf_master,index,worker,is_coroutine = results[1]

        async def get_result(udf_master,index,worker,input_value):
            try:
                res = ray.get(worker.apply.remote(input_value))
            except Exception as inst:
                res = {}
                print(inst)
            finally:    
                ray.get(udf_master.give_back.remote(index))
            return res    

        if is_coroutine:        
            try:
                res = await worker.async_apply.remote(input_value)
            except Exception as inst:
                res = {}
                print(inst)
            finally:    
                ray.get(udf_master.give_back.remote(index)) 
        else:
            res = await get_result(udf_master,index,worker,input_value)            
        
        ray_context.build_result([res])

    @staticmethod
    def _block_apply(ray_context: RayContext):
        conf = ray_context.conf()
        udf_name = conf["UDF_CLIENT"]
        udf_master = ray.get_actor(udf_name)
        [index, worker] = ray.get(udf_master.get.remote())
        input_value = [row["value"] for row in ray_context.python_context.fetch_once_as_rows()]
        try:
            res = ray.get(worker.apply.remote(input_value))
        except Exception as inst:
            res = {}
            print(inst)
        finally:    
            ray.get(udf_master.give_back.remote(index))        
        ray_context.build_result([res])
    
    @staticmethod
    def async_apply(ray_context: RayContext):        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(UDFBuilder._async_apply(ray_context))        

    @staticmethod
    def apply(ray_context: RayContext):                
        # UDFBuilder._block_apply(ray_context)  
        UDFBuilder.async_apply(ray_context)  


class UDFBuildInFunc(object):
    @staticmethod
    def init_tf(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
        from tensorflow.keras import models
        model_path = "./tmp/model/{}".format(str(uuid.uuid4()))
        streaming_tar.save_rows_as_file((ray.get(ref) for ref in model_refs), model_path)
        return models.load_model(model_path)