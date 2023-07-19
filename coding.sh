export LD_LIBRARY_PATH=/home/wangsw/anaconda3/lib:$LD_LIBRARY_PATH && \
python codingfwi.py config/fixed_elastic.yml \
--gpuid 0 \
--opt adam \
--loss vp=niml1_ori vs=niml1_ori rho=niml1_ori \
--mode inversion \
--batchsize 10 \
--lr vp=10.0 vs=5.78 rho=2.89 \
--save-path /mnt/data/wangsw/inversion/elastic_marmousi/results/fixed/aec/test \
--checkpoint /mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss2/l2_newopt/ckpt_9.pt \
--use-cuda \
--grad-cut
# --grad-smooth
