import uuid
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef

from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar
import threading
import asyncio
from ..utils import print_flush

@ray.remote
class UDFMaster(object):
    '''
      UDFMaster is a actor which manage UDFWorkers. 

      num: the number of UDFWorkers
      conf: the configuration setup by !byzerllm
      init_func: init_model in byzer-llm package
      apply_func: stream_chat in byzer-llm package.

      init_func/apply_func will deliver to UDFWorker. 
    '''
    def __init__(self, num: int, conf: Dict[str, str],
                 init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
                 apply_func: Callable[[Any, Any], Any]):
        self.lock = threading.Lock()         
        self.num = num
        self.conf = conf
        self.init_func = init_func
        self.apply_func = apply_func
        # [actor1,actor2,actor3]
        self.actors = []
        # [1,2,3]
        self.actor_indices = []                
        # [4,4,4] concurrency per actor
        self.actor_index_concurrency = []
    
    def workers(self):
        return self.actors.values()
    
    def create_workers(self, conf):
        udf_worker_conf = {}

        if "num_cpus" in conf:
            udf_worker_conf["num_cpus"] = float(conf["num_cpus"])

        infer_backend =  conf.get("infer_backend", "transformers")
        
        if "num_gpus" in conf and not infer_backend.startswith("ray/"):
            udf_worker_conf["num_gpus"] = float(conf["num_gpus"])
        
        if infer_backend.startswith("ray/vllm"):            
            env_vars = {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"true"                    
                        }                                        
            runtime_env = {"env_vars": env_vars}
            udf_worker_conf["runtime_env"] = runtime_env

        if infer_backend.startswith("ray/deepspeed"):    
            udf_worker_conf["num_gpus"] = float(conf["num_gpus"])

        custom_resources = [(key.split("resource.")[1], float(conf[key])) for key in
                            conf.keys() if
                            key.startswith("resource.")]

        if len(custom_resources) > 0:
            udf_worker_conf["resources"] = dict(custom_resources)

        udf_name  = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"
        standalone = conf.get("standalone", "false") == "true"
        model_refs = []
        if "modelServers" in conf and not standalone:
            from .store import transfer_to_ob
            transfer_to_ob(udf_name, conf,model_refs)
        
        workerMaxConcurrency = int(conf.get("workerMaxConcurrency", "1"))
        
        self.actors = dict(
            [(index,
              UDFWorker.options(**udf_worker_conf).remote(model_refs, conf, self.init_func,
                                                          self.apply_func)) for index in
             range(self.num)])
        self.actor_indices = [index for index in range(self.num)]
        self.actor_index_concurrency = [workerMaxConcurrency for _ in range(self.num)]   


    def get(self) -> List[Any]:
        '''
          get a idle UDFWorker to process inference
        '''        
        while sum(self.actor_index_concurrency) == 0:
            time.sleep(0.001)

        # find a idle actor index, the idle actor index in self.actor_index_conurrency should be > 0
        with self.lock:            
            for index in self.actor_indices:
                if self.actor_index_concurrency[index] > 0:
                    self.actor_index_concurrency[index] = self.actor_index_concurrency[index] - 1
                    return [index, self.actors[index]]
        raise Exception("No idle UDFWorker")       

    def give_back(self, index) -> NoReturn:
        '''
          give back a idle UDFWorker
        '''
        with self.lock:
            self.actor_index_concurrency[index] = self.actor_index_concurrency[index] + 1

    def shutdown(self) -> NoReturn:
        [ray.kill(self.actors[index]) for index in self.actor_indices]


@ray.remote
class UDFWorker(object):
    def __init__(self,
                 model_refs: List[ClientObjectRef],
                 conf: Dict[str, str],
                 init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
                 apply_func: Callable[[Any, Any], Any]):
        self.model_refs = model_refs
        self.conf = conf
        self.init_func = init_func
        self.apply_func = apply_func
        self.ready = False
        

    def build_model(self):        
        udf_name  = self.conf["UDF_CLIENT"] if "UDF_CLIENT" in self.conf else "UNKNOW MODEL"
        print_flush(f"MODEL[{udf_name}] Init Model,It may take a while.")
        time1 = time.time()
        self.model = self.init_func(self.model_refs, self.conf)
        time2 = time.time()
        print_flush(f"MODEL[{udf_name}] Successful to init model, time taken:{time2-time1}s") 
        self.ready = True 
        self.model_refs = None      

    def apply(self, v: Any) -> Any:
        if not self.ready:
            udf_name  = self.conf["UDF_CLIENT"] if "UDF_CLIENT" in self.conf else "UNKNOW MODEL"
            raise Exception(f"[{udf_name}] UDFWorker is not ready")
        return self.apply_func(self.model, v)
        
    def is_coroutine(self):        
        return asyncio.iscoroutinefunction(self.apply_func) or asyncio.iscoroutine(self.apply_func) 
    
    async def async_apply(self,v:Any) -> Any:
        if not self.ready:
            udf_name  = self.conf["UDF_CLIENT"] if "UDF_CLIENT" in self.conf else "UNKNOW MODEL"
            raise Exception(f"[{udf_name}] UDFWorker is not ready")
        resp = await self.apply_func(self.model, v)
        return resp

    def shutdown(self):
        ray.actor.exit_actor()


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
