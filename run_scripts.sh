#!/bin/bash

# List of script files to execute
SCRIPTS=(
    ./task1.sh
    ./task2.sh
    ./task3.sh
    ./task4.sh
)

# Loop over scripts and execute them
for script in "${SCRIPTS[@]}"
do
    echo "Executing $script"
    source "$script"
done
