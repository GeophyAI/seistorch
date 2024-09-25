mpirun -f hosts \
python /home/shaowinw/Desktop/wangsw/backup/seistorch/forward.py configs/acoustic_habc_multiple.yml \
--mode forward \
--num-batches 1 \
--use-cuda