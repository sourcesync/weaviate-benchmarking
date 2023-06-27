import datetime
import uuid

import h5py
import numpy as np
import weaviate

#
# Configuration
#

# Path to the numpy base dataset
NPY_PATH = "../../benchmark-data/deep-1M.npy"

# Weaviate connection url
WEAVIATE = "http://localhost:8084"

#
# Tests
#

print("Testing numpy load...")
start_time = datetime.datetime.now()
print("loading", NPY_PATH)
arr = np.load( NPY_PATH )
end_time = datetime.datetime.now()
diff = end_time - start_time
print("numpy load diff=", diff.total_seconds() )

print("Testing array iteration...")
start_time = datetime.datetime.now()
for i in range(arr.shape[0]):
    item = arr[i]
end_time = datetime.datetime.now()
diff = end_time - start_time
print("array iterate (num=%d) diff=" % arr.shape[0], diff.total_seconds() )

print("Testing weaviate batch formation...")
client = weaviate.Client(WEAVIATE, timeout_config=(5, 60))
print("weaviate client=", client)
# remove any previous classes...
client.schema.delete_all()
# Create schema
schema = {
    "classes": [{
        "class": "Benchmark",
        "description": "A class for benchmarking purposes",
        "properties": [
            {
                "dataType": [
                    "int"
                ],
                "description": "The number of the couter in the dataset",
                "name": "counter"
            }
        ],
        "vectorIndexConfig": {
            "ef": -1,
            "efConstruction": 64,
            "maxConnections": 16,
            "vectorCacheMaxObjects": 0,
            "distance": 'cosine'
        }
    }]
}
client.schema.create(schema)
print("Scheme set. Confirming...")
current_schema = client.schema.get()
print("Current schema is", current_schema)
# just create batches, don't add them to index
print("Creating all batches...")
start_time = datetime.datetime.now()
c = 0
batch_c = 0
for i in range(arr.shape[0]):
    vector = arr[i]
    client.batch.add_data_object({
            'counter': c
        },
        'Benchmark',
        str(uuid.uuid3(uuid.NAMESPACE_DNS, str(c))),
        vector = vector
    )
    c += 1
    batch_c += 1
    if ((i%10000)==0):
        print("processed %d/%d" % (i, arr.shape[0]))

end_time = datetime.datetime.now()
diff = end_time - start_time
print("numpy load diff=", diff.total_seconds() )
