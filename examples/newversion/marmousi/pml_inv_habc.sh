torchrun --nproc_per_node=4 /home/shaowinw/seistorch/seistorch_dist.py config/inv_pml.yml  \
--opt adam \
--loss vp=l2 \
--num-batches 4 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/pml_inv_habc \
--grad-cut \
--use-cuda