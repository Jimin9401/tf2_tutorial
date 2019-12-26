from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager


@contextmanager
def poolcontext(*args,**kwargs):
    pool=Pool(*args,**kwargs)
    yield pool
    pool.terminate()
