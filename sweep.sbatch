#!/bin/bash
#SBATCH -p gpu_low
#SBATCH --gres=gpu:1
#SBATCH --mem 30GB 
#SBATCH --time 1-00:00
#SBATCH --exclude=compute-0-19
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dummyblah123@gmail.com # TODO
#SBATCH --chdir=/work/awilf/emtvqa/ # TODO
#SBATCH --output=/work/awilf/emtvqa/logs/%j.out # TODO
#SBATCH --error=/work/awilf/emtvqa/logs/%j.err # TODO

singularity exec -B /work/awilf/ --nv /results/awilf/imgs/tvqa_graph.sif \
wandb agent socialiq/emtvqa/55huy03y
echo "a" >> email_log.txt
