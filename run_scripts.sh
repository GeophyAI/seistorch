#!/bin/bash

# List of script files to execute
SCRIPTS=(
    ./task5.sh
    ./task6.sh
    ./task7.sh
)

# Loop over scripts and execute them
for script in "${SCRIPTS[@]}"
do
    echo "Executing $script"
    source "$script"
done
