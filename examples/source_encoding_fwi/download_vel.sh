rm -rf seismic_inversion
rm -rf velocity_model

git lfs install
# download the velocity model from website
wget https://huggingface.co/datasets/shaowinw/seismic_inversion/blob/main/marmousi_customer/marmousi_20m/true_vp.npy -o ../marmousi_model/true_vp.npy
wget https://huggingface.co/datasets/shaowinw/seismic_inversion/blob/main/marmousi_customer/marmousi_20m/linear_vp.npy -o ../marmousi_model/linear_vp.npy


