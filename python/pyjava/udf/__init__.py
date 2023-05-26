import uuid
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef

from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar
import threading

@ray.remote
class UDFMaster(object):
    def __init__(self, num: int, conf: Dict[str, str],
                 init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
                 apply_func: Callable[[Any, Any], Any]):
        self.lock = threading.Lock()         
        udf_worker_conf = {}

        if "num_cpus" in conf:
            udf_worker_conf["num_cpus"] = float(conf["num_cpus"])

        if "num_gpus" in conf:
            udf_worker_conf["num_gpus"] = float(conf["num_gpus"])

        custom_resources = [(key.split("resource.")[1], float(conf[key])) for key in
                            conf.keys() if
                            key.startswith("resource.")]

        if len(custom_resources) > 0:
            udf_worker_conf["resources"] = dict(custom_resources)
                
        standalone = conf.get("standalone", "false") == "true"
        
        model_refs = []
        print(f"standalone: {standalone} modelServers: {'modelServers' in conf}") 
        if "modelServers" in conf and not standalone:
            model_servers = RayContext.parse_servers(conf["modelServers"])
            print(f"Pull model from {model_servers[0].host}:{model_servers[0].port}")
            time1 = time.time()
            items = RayContext.collect_from(model_servers)
            model_refs = [ray.put(item) for item in items]
            time2 = time.time()
            print(f"Success to pull model, time taken:{time2-time1}s. The total refs: {len(model_refs)}")    
            

        self.actors = dict(
            [(index,
              UDFWorker.options(**udf_worker_conf).remote(model_refs, conf, init_func,
                                                          apply_func)) for index in
             range(num)])
        self._idle_actors = [index for index in range(num)]

    def get(self) -> List[Any]:
        while len(self._idle_actors) == 0:
                time.sleep(0.001)
        with self.lock:            
            index = self._idle_actors.pop()        
        return [index, self.actors[index]]

    def give_back(self, v) -> NoReturn:
        with self.lock:
            self._idle_actors.append(v)

    def shutdown(self) -> NoReturn:
        [ray.kill(self.actors[index]) for index in self._idle_actors]


@ray.remote
class UDFWorker(object):
    def __init__(self,
                 model_refs: List[ClientObjectRef],
                 conf: Dict[str, str],
                 init_func: Callable[[List[ClientObjectRef], Dict[str, str]], Any],
                 apply_func: Callable[[Any, Any], Any]):
        print("Init model....")
        time1 = time.time()
        self.model = init_func(model_refs, conf)
        time2 = time.time()
        print(f"Success to init model. Time taken: {time2 - time1}s")
        self.apply_func = apply_func

    def apply(self, v: Any) -> Any:
        return self.apply_func(self.model, v)

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
        ray_context.build_result([])

    @staticmethod
    def apply(ray_context: RayContext):
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


class UDFBuildInFunc(object):
    @staticmethod
    def init_tf(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
        from tensorflow.keras import models
        model_path = "./tmp/model/{}".format(str(uuid.uuid4()))
        streaming_tar.save_rows_as_file((ray.get(ref) for ref in model_refs), model_path)
        return models.load_model(model_path)
