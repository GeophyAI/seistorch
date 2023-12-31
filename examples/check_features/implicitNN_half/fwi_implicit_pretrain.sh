export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python ../../codingfwi.py config/implicit.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 5 \
--lr vp=1e-5 \
--save-path ./results_implicit_l2 \
--checkpoint ./none \
--use-cuda \
--grad-cut
# --grad-smooth
