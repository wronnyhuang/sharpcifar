from pbs_job_launch import submit_job

opts = {   "job_name":"hess",
            "hours":40,
           "mins":3,
           "secs":0,
           "account":"MHPCC96670DA1",   #"USNAM37766Z80",
           "num_nodes":1,
           "procs_per_node":1,
           "cpus_per_node":20,
           "queue":"standard",
           "delete_after_submit":False,  #  Delete the pbs shell script immediately after calling qsub?
           "call_qsub":True
        }


opts["setup_command"] = \
"""
cdw
cd /gpfs/scratch/tomg/sharpcifar
"""

opts["outfile_prefix"] = "/gpfs/scratch/tomg/sharpcifar/launch/pbs_scripts/"


with open('jobs3.txt') as f:
    content = f.readlines()


content = [c.strip() for c in content if 'python' in c]
gpu0 = [c for c in content if 'gpu=0' in c]
gpu1 = [c for c in content if 'gpu=1' in c]
gpu2 = [c for c in content if 'gpu=2' in c]
gpu3 = [c for c in content if 'gpu=3' in c]

jobs = zip(gpu0,gpu1,gpu2,gpu3)


for i,job in enumerate(jobs):
    command = """
    %s  &> log/console3_gpu=0_job=%d.out &
    pid0=$!
    %s  &> log/console3_gpu=1_job=%d.out &
    pid1=$!
    %s  &> log/console3_gpu=2_job=%d.out &
    pid2=$!
    %s  &> log/console3_gpu=3_job=%d.out &
    pid3=$!
    wait $pid0
    wait $pid1
    wait $pid2
    wait $pid3
    """ % ( job[0], i, job[1], i, job[2], i, job[3], i)

    opts["job_name"] = 'sharp' + str(i)

    submit_job(command, **opts)

