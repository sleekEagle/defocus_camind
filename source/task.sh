#!/bin/bash
#SBATCH --job-name="camind"
#SBATCH --error="/p/blurdepth/results/camind_defdata/camind.err"
#SBATCH --output="/p/blurdepth/results/camind_defdata/camind.out"
#SBATCH --partition="gpu"
#SBATCH -w jaguar02
python train.py --lr 0.0001 --datapath /p/blurdepth/data/nyu_depth/ --savepath /p/blurdepth/models/camind/camind_nyu --checkpt /p/blurdepth/models/camind/camind_nyu/camind_nyu_28.0_blurclip65.0_blurweight1.0/999.pth
