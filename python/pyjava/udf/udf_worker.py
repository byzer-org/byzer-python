import uuid
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import asyncio
from pyjava.utils import print_flush
import threading


@ray.remote
class UDFWorker(object):
    def __init__(self,
                 model_refs: List[ClientObjectRef],
                 conf: Dict[str, str],
                 init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
                 apply_func: Callable[[Any, Any], Any]):
        self.lock = threading.Lock()
        self.model_refs = model_refs
        self.conf = conf
        self.init_func = init_func
        self.apply_func = apply_func
        self.ready = False
        self.active_task = 0


    def _inc_active_task(self):
        with self.lock:
            self.active_task += 1    

    def _dec_active_task(self):
        with self.lock:
            self.active_task -= 1    

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
        self._inc_active_task()
        try:
            return self.apply_func(self.model, v)
        finally:
            self._dec_active_task()

        
    def is_coroutine(self):        
        return asyncio.iscoroutinefunction(self.apply_func) or asyncio.iscoroutine(self.apply_func) 
    
    async def async_apply(self,v:Any) -> Any:
        if not self.ready:
            udf_name  = self.conf["UDF_CLIENT"] if "UDF_CLIENT" in self.conf else "UNKNOW MODEL"
            raise Exception(f"[{udf_name}] UDFWorker is not ready")
        self._inc_active_task()
        try:
            resp = await self.apply_func(self.model, v)
            return resp
        finally:
            self._dec_active_task()

    async def stat(self):
        return {
            "active_task": self.active_task,
            "ready": self.ready
        }


    def shutdown(self):
        ray.actor.exit_actor()