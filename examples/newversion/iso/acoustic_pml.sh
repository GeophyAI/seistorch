mpirun -f hosts \
python /home/shaowinw/Desktop/wangsw/backup/seistorch/forward.py configs/acoustic_pml.yml \
--mode forward \
--num-batches 1 \
--use-cuda