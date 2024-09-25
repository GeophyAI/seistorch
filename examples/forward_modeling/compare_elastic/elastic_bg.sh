mpirun -f hosts \
python ../../../forward.py configs/elastic_bg.yml \
--mode forward \
--num-batches 5 \
--use-cuda