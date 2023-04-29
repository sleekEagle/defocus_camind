#!/bin/bash
#SBATCH --job-name="camind_f1"
#SBATCH --error="/p/blurdepth/results/camind_defdata/camind_f1.err"
#SBATCH --output="/p/blurdepth/results/camind_defdata/camind_f1.out"
#SBATCH --partition="gpu"
#SBATCH -w cheetah03
python train.py --lr 0.0001 --datapath /p/blurdepth/data/nyu_depth/ --savepath /p/blurdepth/models/camind/camind_nyu
