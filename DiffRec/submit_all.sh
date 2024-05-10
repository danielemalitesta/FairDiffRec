#!/bin/bash

first_script_id=$(sbatch ./scripts/$1/*__$2.sh | awk '{print $NF}')

for i in {$($2+1)..$3}; do
    script="./scripts/$1/*__${i}.sh"
    # Submit the script with dependency on the previous script
    script_id=$(sbatch --dependency=afterok:${first_script_id} ${script} | awk '{print $NF}')
    # Update the dependency for the next iteration
    first_script_id=$script_id
done