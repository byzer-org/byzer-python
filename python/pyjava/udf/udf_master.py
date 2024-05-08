import uuid
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientObjectRef

import threading
import numpy as np
from pyjava.udf.udf_worker import UDFWorker

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
        self.actor_index_update_time = np.array([])
        self.request_count = []
    
    def workers(self):
        return self.actors.values()
    
    def create_worker_conf(self, conf):
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

        return udf_worker_conf    
    
    def create_worker(self, index, conf):    
        udf_worker_conf = self.create_worker_conf(conf)        
        udf_name  = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"
        standalone = conf.get("standalone", "false") == "true"
        model_refs = []
        if "modelServers" in conf and not standalone:
            from .store import transfer_to_ob
            transfer_to_ob(udf_name, conf,model_refs)
        
        return UDFWorker.options(**udf_worker_conf).remote(model_refs, conf, self.init_func,
                                                           self.apply_func)

    def create_workers(self, conf):
        workerMaxConcurrency = int(conf.get("workerMaxConcurrency", "1"))
        
        self.actors = dict(
            [(index, self.create_worker(index, conf)) for index in range(self.num)])

        self.load_balance = conf.get("load_balance", "lru").lower()

        ## LRU    
        self.actor_indices = [index for index in range(self.num)]
        self.actor_index_concurrency = [workerMaxConcurrency for _ in range(self.num)] 
        self.actor_index_update_time = np.array([time.monotonic() for _ in range(self.num)])    

        ## Round Robin 
        self.request_count = [0 for _ in range(self.num)]
        self.counter = 0

    def get_worker_max_concurrency(self):
        return int(self.conf.get("workerMaxConcurrency", "1"))    

    def get(self,index = -1 ) -> List[Any]:
        '''
          get a idle UDFWorker to process inference
        '''
        if index != -1:
            self.request_count[index] += 1
            return [index, self.actors[index]]   

        if self.load_balance == "round_robin":
            with self.lock: 
                index = self.counter
                self.counter = (self.counter + 1) % self.num
                self.request_count[index]+= 1
            return [index, self.actors[index]]        

        # find a idle actor index, the idle actor index in self.actor_index_conurrency should be > 0
        with self.lock:
            retry = int(self.conf.get("timeout", 10))/0.001
            while True: 
                ## find the worker who has not be visited for the longest time
                index = np.argmin(self.actor_index_update_time)                                
                if self.actor_index_concurrency[index] > 0:
                        self.actor_index_concurrency[index] = self.actor_index_concurrency[index] - 1
                        self.actor_index_update_time[index] = time.monotonic()
                        self.request_count[index] += 1
                        return [index, self.actors[index]]
                else:  
                    actor = self.actors[index]
                    v = ray.get(actor.stat.remote())                    
                    ## the worker maybe leak, reset the worker
                    if v["active_task"] == 0 or time.monotonic() - self.actor_index_update_time[index] > 10*60:                        
                        self.actor_index_concurrency[index] = self.get_worker_max_concurrency()
                        self.actor_index_update_time[index] = time.monotonic()
                    ## find the worker who left the most concurrencies 
                    index = np.argmax(self.actor_index_concurrency)
                    if self.actor_index_concurrency[index] > 0:
                        self.actor_index_concurrency[index] = self.actor_index_concurrency[index] - 1
                        self.actor_index_update_time[index] = time.monotonic()
                        self.request_count[index] += 1
                        return [index, self.actors[index]]

                    if retry > 0:                        
                        time.sleep(0.001)                        
                        retry = retry - 1                        
                    else:
                        break
                
        raise Exception("No idle UDFWorker") 

    def reset(self):        
        workerMaxConcurrency = int(self.conf.get("workerMaxConcurrency", "1"))
        self.actor_indices = [index for index in range(self.num)]
        self.actor_index_concurrency = [workerMaxConcurrency for _ in range(self.num)] 
        self.actor_index_update_time = np.array([time.monotonic() for _ in range(self.num)]  )
        self.counter = 0

    def give_back(self, index) -> NoReturn:
        '''
          give back a idle UDFWorker
        '''
        with self.lock:
            if self.actor_index_concurrency[index] < self.get_worker_max_concurrency():
                self.actor_index_concurrency[index] = self.actor_index_concurrency[index] + 1
            # self.actor_index_update_time[index] = time.monotonic()

    def shutdown(self) -> NoReturn:
        [ray.kill(self.actors[index]) for index in self.actor_indices]
        
    def stat(self) -> Dict[str, Any]:
        '''
          Show the current status of UDFMaster
        '''
        busy_workers = sum(1 for concurrency in self.actor_index_concurrency if concurrency < int(self.conf.get("workerMaxConcurrency", "1")))
        idle_workers = self.num - busy_workers

        worker_states = []
        for actor in self.actors.values():
            v = ray.get(actor.stat.remote())
            worker_states.append(v)
        
        return {
            "total_workers": self.num,
            "busy_workers": busy_workers,  
            "idle_workers": idle_workers,
            "load_balance_strategy": self.load_balance,
            "total_requests": self.request_count,
            "state": self.actor_index_concurrency,
            "worker_max_concurrency": self.conf.get("workerMaxConcurrency", "1"),
            "workers_last_work_time": [f"{time.monotonic() - self.actor_index_update_time[index]}s" for index in range(self.num)],
            "worker_state": worker_states,
        }

    def reload_worker(self, index):
        '''
        reload a worker
        '''
        if "modelServers" in self.conf:
            raise Exception("reload_worker is not supported when using model server")
        
        # create a new worker
        new_worker = self.create_worker(index, self.conf)
        # load model in new worker
        ray.get(new_worker.build_model.remote())
        
        # replace the old worker with new worker
        old_worker = self.actors[index]
        self.actors[index] = new_worker
        
        # shutdown the old worker
        ray.kill(old_worker)

