import torch
import torch.cuda
import numpy as np
import random
from .logger import get_logger
import torch.multiprocessing as mp
import threading
import time

def run_exps(devices, jobs, block=True, interval=0.1):
    '''
    jobs:
      - func
      - kwargs
      - callback
    '''
    job_queue = jobs
    dev_queue = devices
    
    pool = mp.Pool()

    def gen_callback(dev, calls):
        def callback(res):
            dev_queue.put(dev)
            return calls(res)
        return callback

    def gen_err_callback(dev, func):
        def callback(err):
            dev_queue.put(dev)
            return func(err)
        return callback

    def run_job():
        while True:
            job = job_queue.get()
            if job == False:
                break
            dev = dev_queue.get()
            pool.apply_async(
                job['func'],
                kwds={**job['kwargs'], 'device': dev},
                callback=gen_callback(dev, job['callback']),
                error_callback=gen_err_callback(dev, job['err'] if 'err' in job else lambda x: x)
            )
            time.sleep(interval)
    
    th = threading.Thread(target=run_job)
    th.daemon = True
    th.start()
    if block:
        th.join()
        pool.close()
        pool.join()
    return th, pool

def set_seed(seed=2020):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)

__all__ = ['set_seed', 'get_logger', 'run_exps']
