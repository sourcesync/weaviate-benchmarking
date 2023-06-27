#!/bin/sh

#
# You need to change these parameters to reflect your local setup
#

# Write/append the benchmark results to this file
DT=$(date +%s)
OUTPUT="/tmp/$(echo $DT)_DATA_$(echo $RAN | md5sum | head -c 20).txt"
LOG="/tmp/$(echo $DT)_LOG_$(echo $RAN | md5sum | head -c 20).txt"
echo "output=$OUTPUT"
echo "log=$LOG"

# Set a valid allocation id
ALLOCATION_ID="fd283b38-3e4a-11eb-a205-7085c2c5e516" #"0b391a1a-b916-11ed-afcb-0242ac1c0002" #"fd283b38-3e4a-11eb-a205-7085c2c5e516"

# Path to dataset numpy file
DATASET="/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy"

# Path to queries numpy file
QUERIES="/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"

# Path to the ground truth numpy file
GROUNDTRUTH="/mnt/nas1/fvs_benchmark_datasets/deep-10K-gt-10.npy"

# Set Swagger to emit verbose http payload messages
export GEMINI_SWAGGER_VERBOSE='true'

#
# Check the arguments and the system before running the benchmarks.
#
if [ -f "$OUTPUT" ]; then
	echo "Error:  The file $OUTPUT already exists.  Please move that file or use a file path that does not already exist."
	exit 1
fi

# Make sure this script exists on any error going forward.
set -e

#
# Now run all the benchmarks.  Note that this might take a while, so you should consider running it behind the 'screen' utility.
#
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --save --b 64 2>&1 | tee "$LOG"

echo "Done. Output=$OUTPUT, Log=$LOG"
