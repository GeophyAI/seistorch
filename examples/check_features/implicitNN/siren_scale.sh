export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../codingfwi.py config/siren_scale.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 10 \
--lr vp=0.001 \
--save-path ./results_implicit_sirenscale \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
