import os
import random
import subprocess

import numpy as np


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    SLURM_VARIABLES = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_TASK_PID",
    ]

    PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
    for name in SLURM_VARIABLES:
        value = os.environ.get(name, None)
        print(PREFIX + "%s: %s" % (name, str(value)))

    # number of nodes / node ID
    params.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    params.node_rank = int(os.environ["SLURM_NODEID"])

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    params.master_addr = hostnames.split()[0].decode("utf-8")
    print("master address ", params.master_addr)
