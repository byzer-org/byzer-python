import uuid
import asyncio
import time
from typing import Any, NoReturn, Callable, Dict, List
import ray
from pyjava.api.mlsql import RayContext
from ..utils import print_flush

async def worker(queue,udf_name,model_refs):
    count = 0
    while True:
        item = await queue.get()
        if item is None:
            # Signal to exit when queue is empty
            break
        if count % 1000 == 0:
            print_flush(f"MODEL[{udf_name}] UDFMaster push model: {count}")
        count += 1  
        model_refs.append(ray.put(item))

async def producer(items,udf_name, queue):    
    for item in items:        
        await queue.put(item)
    await queue.put(None)        

async def _transfer_to_ob(udf_name, conf,model_refs):     
    model_servers = RayContext.parse_servers(conf["modelServers"]) 
    udf_name  = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"

    print_flush(f"MODEL[{udf_name}] Transfer model from {model_servers[0].host}:{model_servers[0].port} to Ray Object Store")                       
    time1 = time.time()
    queue = asyncio.Queue(1000)
    worker_task = asyncio.create_task(worker(queue,udf_name,model_refs))
    producer_task = asyncio.create_task(producer(RayContext.collect_from(model_servers), udf_name,queue))    
    await asyncio.gather(producer_task,worker_task)
    print_flush(f"MODEL[{udf_name}] UDFMaster push model to object store cost {time.time() - time1} seconds") 

def block_transfer_to_ob(udf_name,conf,model_refs):
    
    model_servers = RayContext.parse_servers(conf["modelServers"]) 
    print_flush(f"MODEL[{udf_name}] Transfer model from {model_servers[0].host}:{model_servers[0].port} to Ray Object Store")                       
    time1 = time.time()    
    udf_name  = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"
    
    for item in RayContext.collect_from(model_servers):
        model_refs.append(ray.put(item)) 
    
    print_flush(f"MODEL[{udf_name}] UDFMaster push model to object store cost {time.time() - time1} seconds")            

    

def transfer_to_ob(udf_name, conf,model_refs):
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(_transfer_to_ob(udf_name, conf,model_refs))                
    block_transfer_to_ob(udf_name, conf,model_refs)