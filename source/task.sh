#!/bin/bash
#SBATCH --job-name="camind"
#SBATCH --error="/p/blurdepth/results/camind_defdata/camind.err"
#SBATCH --output="/p/blurdepth/results/camind_defdata/camind.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah01
python train.py
