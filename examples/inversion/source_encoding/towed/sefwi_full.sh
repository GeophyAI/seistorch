export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../../codingfwi.py full.yml \
--gpuid 0 \
--opt steepestdescent \
--loss vp=cs \
--mode inversion \
--batchsize 87 \
--lr vp=10.0 \
--save-path ./results_full \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
