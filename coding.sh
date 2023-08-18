export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/test_source_inversion2/obs.yml \
--gpuid 0 \
--opt adam \
--loss vp=l2 \
--mode inversion \
--batchsize 20 \
--lr vp=10.0 \
--save-path /public1/home/wangsw/FWI/EFWI/Marmousi/Marmousi2/results/diff_wavelet/l2_true \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
