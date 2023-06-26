import datetime

import h5py
import numpy as np

# Path to the numpy base dataset
NPY_PATH = "../../benchmark-data/deep-1M.npy"

start_time = datetime.datetime.now()

print("loading", NPY_PATH)

arr = np.load( NPY_PATH )

end_time = datetime.datetime.now()

diff = end_time - start_time

print("diff=", diff.total_seconds() )





