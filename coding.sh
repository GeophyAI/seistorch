export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py /public1/home/wangsw/FWI/NO_LOWFREQ/config/02b_acoustic_invert.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path /home/ \
--checkpoint /home/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
