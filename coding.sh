export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py /public1/home/wangsw/FWI/NO_LOWFREQ/config/02_acoustic_invert_p.yml \
--gpuid 0 \
--opt adam \
--loss vp=fa \
--mode inversion \
--batchsize 1 \
--lr vp=10.0 \
--save-path /public1/home/wangsw/FWI/NO_LOWFREQ/fa \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
