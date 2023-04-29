#!/bin/bash
#SBATCH --job-name="camind_fl100"
#SBATCH --error="/p/blurdepth/results/camind_defdata/camind_fl100.err"
#SBATCH --output="/p/blurdepth/results/camind_defdata/camind_fl100.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar02
python train.py --lr 0.0001 --datapath /p/blurdepth/data/nyu_depth/ --savepath /p/blurdepth/models/camind/camind_nyu
