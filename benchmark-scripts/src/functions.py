import os
import uuid
import json
import time
import datetime
import subprocess
import h5py
import weaviate
import loguru
import numpy as np

def add_batch(client, c, vector_len):
    '''Adds batch to Weaviate and returns
       the time it took to complete in seconds.'''

    start_time = datetime.datetime.now()
    results = client.batch.create_objects()
    stop_time = datetime.datetime.now()
    handle_results(results)
    run_time = stop_time - start_time
    if (c % 10000) == 0:
        loguru.logger.info('Import status => added ' + str(c) + ' of ' + str(vector_len) + ' objects')
    return run_time.seconds


def handle_results(results):
    '''Handle error message from batch requests
       logs the message as an info message.'''
    if results is not None:
        for result in results:
            if 'result' in result and 'errors' in result['result'] and  'error' in result['result']['errors']:
                for message in result['result']['errors']['error']:
                    loguru.logger.error(message['message'])



def run_speed_test(l, CPUs,weaviate_url):
    '''Runs the actual speed test in Go'''
    process = subprocess.Popen(['./benchmarker','dataset', '-u', weaviate_url, '-c', 'Benchmark', '-q', 'queries.json', '-p', str(CPUs), '-f', 'json', '-l', str(l)], stdout=subprocess.PIPE)
    result_raw = process.communicate()[0].decode('utf-8')
    return json.loads(result_raw)


def conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections):
    '''Conducts the benchmark, note that the NN results
       and speed test run seperatly from each other'''

    # result obj
    results = {
        'benchmarkFile': benchmark_file[0],
        'distanceMetric': benchmark_file[1],
        'totalTested': 0,
        'ef': ef,
        'efConstruction': efConstruction,
        'maxConnections': maxConnections,
        'requestTimes': {}
    }

    # update schema for ef setting
    loguru.logger.info('Update "ef" to ' + str(ef) + ' in schema')
    client.schema.update_config('Benchmark', { 'vectorIndexConfig': { 'ef': ef } })
    c = 0
    loguru.logger.info('Find neighbors with ef = ' + str(ef))
    print("file", benchmark_file[0])
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        tmp = []
        test_vectors = f['test']
        test_vectors_len = len(f['test'])
        for test_vector in test_vectors:

            # set certainty for  l2-squared
            nearVector = { "vector": test_vector.tolist() }
            
            # Start request
            query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(10).do()    
            inds = [x["counter"] for x in query_result["data"]["Get"]["Benchmark"]]
            tmp.append(inds)
            # log ouput
            if (c % 1000) == 0:
                loguru.logger.info('Validated ' + str(c) + ' of ' + str(test_vectors_len))

            c+=1
        inds = np.array(tmp)
        np.save(f"/results/test_inds_{ef}_{inds.shape}.npy")

    ##
    # Run the speed test
    ##
    loguru.logger.info('Run the speed test')
    train_vectors_len = 0
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        train_vectors_len = len(f['train'])
        test_vectors_len = len(f['test'])
        vector_write_array = []
        for vector in f['test']:
            vector_write_array.append(vector.tolist())
        with open('queries.json', 'w', encoding='utf-8') as jf:
            json.dump(vector_write_array, jf, indent=2)
        results['requestTimes']['limit_1'] = run_speed_test(1, CPUs, weaviate_url)
        results['requestTimes']['limit_10'] = run_speed_test(10, CPUs, weaviate_url)
        results['requestTimes']['limit_100'] = run_speed_test(100, CPUs, weaviate_url)

    # add final results
    results['totalTested'] = c
    results['totalDatasetSize'] = train_vectors_len
    # for k in ['1', '10', '100']:
    #     results['recall'][k]['average'] = sum(all_scores[k]) / len(all_scores[k])

    return results


def remove_weaviate_class(client):
    '''Removes the main class and tries again on error'''
    try:
        client.schema.delete_all()
        # Sleeping to avoid load timeouts
    except:
        loguru.logger.exception('Something is wrong with removing the class, sleep and try again')
        time.sleep(240)
        remove_weaviate_class(client)


def import_into_weaviate(client, efConstruction, maxConnections, benchmark_file, curr, size):
    '''Imports the data into Weaviate'''
    
    # variables
    benchmark_import_batch_size = 10000
    benchmark_class = 'Benchmark'
    import_time = 0

    # TODO: "DO NOT" Delete schema if available
    current_schema = client.schema.get()
    if len(current_schema['classes']) > 0 and curr > 0:
        remove_weaviate_class(client)
    else:
        schema = {
            "classes": [{
                "class": benchmark_class,
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
                    "efConstruction": efConstruction,
                    "maxConnections": maxConnections,
                    "vectorCacheMaxObjects": 1000000000,
                    "distance": benchmark_file[1]
                }
            }]
        }

        client.schema.create(schema)

    # Import
    loguru.logger.info('Start import process for ' + benchmark_file[0] + ', ef' + str(efConstruction) + ', maxConnections' + str(maxConnections))
    import os
    print("FUNC h5py", benchmark_file[0], os.listdir("/var/hdf5") )
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        vectors = f['train'][curr:size]
        c = 0
        batch_c = 0
        vector_len = len(vectors)
        for vector in vectors:
            client.batch.add_data_object({
                    'counter': c
                },
                'Benchmark',
                str(uuid.uuid3(uuid.NAMESPACE_DNS, str(c))),
                vector = vector
            )
            if batch_c == benchmark_import_batch_size:
                import_time += add_batch(client, c, vector_len)
                batch_c = 0
            c += 1
            batch_c += 1
        import_time += add_batch(client, c, vector_len)
    loguru.logger.info('done importing ' + str(c) + ' objects in ' + str(import_time) + ' seconds')

    return import_time


def run_the_benchmarks(weaviate_url, CPUs, efConstruction_array, maxConnections_array, ef_array, benchmark_file_array, size, increment, stop):
    '''Runs the actual benchmark.
       Results are stored in a JSON file'''

    # Connect to Weaviate Weaviate
    try:
        client = weaviate.Client(weaviate_url, timeout_config=(5, 60))
    except:
        print('Error, can\'t connect to Weaviate, is it running?')
        exit(1)

    client.batch.configure(
        timeout_retries=10,
    )
    curr = 0

    # itterate over settings
    efConstruction = efConstruction_array[0]
    maxConnections = maxConnections_array[1]
    benchmark_file = benchmark_file_array[0]
    ef = ef_array[0]
    while curr < stop:
        # import data
        print("before import", benchmark_file)
        import_time = import_into_weaviate(client, efConstruction, maxConnections, benchmark_file, curr, size)
        curr, size = size, size + increment
        # Find neighbors based on UUID and ef settings
        results = []
        result = conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections)
        result['importTime'] = import_time
        result["dataSize"] = curr
        results.append(result)
        
        # write json file
        if not os.path.exists('results'):
            os.makedirs('results')
        output_json = 'results/weaviate_benchmark' + '__' + benchmark_file[0] + '__' + str(efConstruction) + '__' + str(maxConnections) + '.json'
        loguru.logger.info('Writing JSON file with results to: ' + output_json)
        with open(output_json, 'w') as outfile:
            json.dump(results, outfile)

    loguru.logger.info('completed')
