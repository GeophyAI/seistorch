mpirun -f hosts \
python ../../../forward.py configs/elastic.yml \
--mode forward \
--num-batches 5 \
--use-cuda