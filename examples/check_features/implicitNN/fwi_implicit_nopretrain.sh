export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../../codingfwi.py config/implicit_nopretrain.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 5 \
--lr vp=0.0001 \
--save-path ./results_implicit_np \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
