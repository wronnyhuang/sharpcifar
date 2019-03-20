
#####################################
#   PBS script auto generated
#   by pbs_job_launch.py
#####################################
#!/bin/sh -f
#$ -cwd
#PBS -N sharp11
#PBS -o /gpfs/scratch/tomg/sharpcifar/launch/pbs_scripts/sharp11_output_UvR7OwSeF.txt
#PBS -j oe
#PBS -l walltime=20:3:0
#PBS -A MHPCC96670DA1
#PBS -l select=1:mpiprocs=1:ncpus=20
#PBS -q standard
#####################################
echo "-----------------command run by pbs_job_launch.py-----------------------"
echo "
    python surface2d.py -name=svhn_clean  -gpu=0 -part=20 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=0_job=11.out &
    pid0=$!
    python surface2d.py -name=svhn_clean  -gpu=1 -part=21 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=1_job=11.out &
    pid1=$!
    python surface2d.py -name=svhn_clean  -gpu=2 -part=22 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=2_job=11.out &
    pid2=$!
    python surface2d.py -name=svhn_clean  -gpu=3 -part=23 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=3_job=11.out &
    pid3=$!
    wait $pid0
    wait $pid1
    wait $pid2
    wait $pid3
    "
echo "---------------------------pbs options----------------------------------"
echo PBS -l select=1:mpiprocs=1:ncpus=20
echo PBS -l walltime=20:3:0
echo PBS -A MHPCC96670DA1
echo PBS -q standard
echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo "
cdw
cd /gpfs/scratch/tomg/sharpcifar
"

cdw
cd /gpfs/scratch/tomg/sharpcifar

echo "
    python surface2d.py -name=svhn_clean  -gpu=0 -part=20 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=0_job=11.out &
    pid0=$!
    python surface2d.py -name=svhn_clean  -gpu=1 -part=21 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=1_job=11.out &
    pid1=$!
    python surface2d.py -name=svhn_clean  -gpu=2 -part=22 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=2_job=11.out &
    pid2=$!
    python surface2d.py -name=svhn_clean  -gpu=3 -part=23 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=3_job=11.out &
    pid3=$!
    wait $pid0
    wait $pid1
    wait $pid2
    wait $pid3
    "

    python surface2d.py -name=svhn_clean  -gpu=0 -part=20 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=0_job=11.out &
    pid0=$!
    python surface2d.py -name=svhn_clean  -gpu=1 -part=21 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=1_job=11.out &
    pid1=$!
    python surface2d.py -name=svhn_clean  -gpu=2 -part=22 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=2_job=11.out &
    pid2=$!
    python surface2d.py -name=svhn_clean  -gpu=3 -part=23 -npart=32 -url=https://www.dropbox.com/sh/v1unz2j931fsmkr/AADVteaZ51B5Xy0eEsMzlgDWa?dl=0  > log/console_gpu=3_job=11.out &
    pid3=$!
    wait $pid0
    wait $pid1
    wait $pid2
    wait $pid3
    
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"

exit
