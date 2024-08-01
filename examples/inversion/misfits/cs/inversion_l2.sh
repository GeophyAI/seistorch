torchrun --nproc_per_node=1 --master-port 12306 \
../../../../seistorch_dist.py configs/inversion.yml \
--opt adam \
--loss vp=l2 \
--lr vp=20.0 \
--mode inversion \
--save-path ./results/towed_l2 \
--filteratfirst \
--grad-cut \
--use-cuda