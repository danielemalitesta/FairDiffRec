#!/bin/bash

first_script_id=$(sbatch *__$1.sh | awk '{print $NF}')

for i in {$1+1..$2}; do
    script="*__${i}.sh"
    # Submit the script with dependency on the previous script
    script_id=$(sbatch --dependency=afterok:${first_script_id} ${script} | awk '{print $NF}')
    # Update the dependency for the next iteration
    first_script_id=$script_id
done