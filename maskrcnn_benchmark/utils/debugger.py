import ipdb

from maskrcnn_benchmark.config import cfg

def set_trace():
    if cfg.DEBUG:
        ipdb.set_trace(context=10)