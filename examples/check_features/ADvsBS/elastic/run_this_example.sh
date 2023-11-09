echo 'Cleaning the old files'
sh clean.sh

echo 'Generate the geometry and velocity'
python3 generate_model_geometry.py
echo 'Done.'

echo 'Running forward modeling'
sh forward.sh
echo 'Done.'

echo 'Running FWI with automatic differentiation'
sh inversion_AD.sh
echo 'Done.'

echo 'Running FWI with boundary saving-based automatic differentiation'
sh inversion_BS.sh
echo 'Done.'

echo 'Compare the results of AD and BS.'
python3 AD_vs_BS.py
echo 'Done.'
