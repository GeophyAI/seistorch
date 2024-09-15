echo 'Cleaning the old files'
sh clean.sh

echo 'Generate the geometry and velocity'
python3 generate_model_geometry.py
echo 'Done.'

echo 'Running forward modeling'
sh forward.sh
echo 'Done.'

echo 'Running FWI with AD and HABC'
sh inversion_ADHABC.sh
echo 'Done.'

echo 'Running FWI with BS and HABC'
sh inversion_BSHABC.sh
echo 'Done.'

echo 'Running FWI with BS and PML'
sh inversion_BSPML.sh
echo 'Done.'

echo 'Running FWI with AD and PML'
sh inversion_ADPML.sh
echo 'Done.'

echo 'Compare the results of AD and BS.'
python3 AD_vs_BS.py
echo 'Done.'
