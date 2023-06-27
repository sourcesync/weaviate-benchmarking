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

def size_to_str(size):
    tmp = len(str(size))
    if tmp < 7:
        data_name = f"{int(size/1000)}K"
    elif tmp < 10:
        data_name = f"{int(size/1000000)}M"
    else:
        data_name = f"{int(size/1000000000)}B"
    return data_name


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


def match_results(test_set, weaviate_result_set, k, size, ef):
    '''Match the reults from Weaviate to the benchmark data.
       If a result is in the returned set, score goes +1.
       Because there is checked for 100 neighbors a score
       of 100 == perfect'''

    # set score
    score = 0

    # return if no result
    if weaviate_result_set['data']['Get']['Benchmark'] == None:
        return score

    # create array from Weaviate result
    weaviate_result_array = []
    for weaviate_result in weaviate_result_set['data']['Get']['Benchmark'][:k]:
        weaviate_result_array.append(weaviate_result['counter'])
    inds = np.array(weaviate_result_array)
    np.save(f"results/inds-deep-{size_to_str(size)}-{ef}.npy", inds)

    # match scores
    for nn in test_set[:k]:
        if nn in weaviate_result_array:
            score += 1
    
    return score


def run_speed_test(l, CPUs,weaviate_url):
    '''Runs the actual speed test in Go'''
    process = subprocess.Popen(['./benchmarker','dataset', '-u', weaviate_url, '-c', 'Benchmark', '-q', 'queries.json', '-p', str(CPUs), '-f', 'json', '-l', str(l)], stdout=subprocess.PIPE)
    result_raw = process.communicate()[0].decode('utf-8')
    print("raw speed test result: ", result_raw)
    return json.loads(result_raw)


def conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections, size, multi):
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
        'recall': {
#            '100': {
#                'highest': 0,
#                'lowest': 100,
#                'average': 0
#            },
            '10': {
                'highest': 0,
                'lowest': 100,
                'average': 0
            },
#            '1': {
#                'highest': 0,
#                'lowest': 100,
#                'average': 0
#            },
        },
        'requestTimes': {}
    }

    # update schema for ef setting
    loguru.logger.info('Update "ef" to ' + str(ef) + ' in schema')
    client.schema.update_config('Benchmark', { 'vectorIndexConfig': { 'ef': ef } })

    ##
    # Run the score test
    ##
    c = 0
    all_scores = {
#            '100':[],
            '10':[],
##            '1': [],
        }

    loguru.logger.info('Find neighbors with ef = ' + str(ef))
    #print("file", benchmark_file[0])
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        test_vectors = f['test']
        test_vectors_len = len(f['test'])
        for test_vector in test_vectors:

            # set certainty for  l2-squared
            nearVector = { "vector": test_vector.tolist() }
            
            # Start request
            query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(100).do()    
            #print("qr", query_result)
            for k in [10]: #[1, 10,100]:
                k_label=f'{k}'
                #print("k", f['neighbors'][c])
                score = match_results(f['neighbors'][c], query_result, k, size, ef)
                # if score == 0:
                    # loguru.logger.info('There is a 0 score, this most likely means there is an issue with the dataset OR you have very low index settings. Found for vector: ' + str(test_vector[0]))
                # all_scores[k_label].append(score)
                
                # # set if high and low score
                # if score > results['recall'][k_label]['highest']:
                #     results['recall'][k_label]['highest'] = score
                # if score < results['recall'][k_label]['lowest']:
                #     results['recall'][k_label]['lowest'] = score

            # log ouput
            if (c % 1000) == 0:
                loguru.logger.info('Validated ' + str(c) + ' of ' + str(test_vectors_len))

            c+=1

    ##
    # Run the speed test
    ##
    loguru.logger.info('Run the speed test')
    train_vectors_len = 0
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        train_vectors_len = len(f['train'][:size])
        test_vectors_len = len(f['test'])
        # print(train_vectors_len, test_vectors_len)
        vector_write_array = []
        for vector in f['test']:
            vector_write_array.append(vector.tolist())
        with open('queries.json', 'w', encoding='utf-8') as jf:
            json.dump(vector_write_array, jf, indent=2)
        #results['requestTimes']['limit_1'] = run_speed_test(1, CPUs, weaviate_url)
        print("vector write array: ", vector_write_array)
        results['requestTimes']['limit_10'] = run_speed_test(10, CPUs, weaviate_url)
        #results['requestTimes']['limit_100'] = run_speed_test(100, CPUs, weaviate_url)
        results['totalDatasetSize'] = train_vectors_len
    if not multi:
        # add final results
        results['totalTested'] = c
        for k in [ '10']: #['1', '10', '100']:
            results['recall'][k]['average'] = sum(all_scores[k]) / len(all_scores[k])

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

    # Delete schema if available
    current_schema = client.schema.get()
    if curr == 0:
        if len(current_schema['classes']) > 0:
            remove_weaviate_class(client)

        # Create schema
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
                str(uuid.uuid3(uuid.NAMESPACE_DNS, str(c+curr))),
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


def run_the_benchmarks(weaviate_url, CPUs, efConstruction_array, maxConnections_array, ef_array, benchmark_file_array, multi, size, increment, stop):
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
    if not multi:
        stop = 0
    # itterate over settings
    while curr <= stop:
        for benchmark_file in benchmark_file_array:
            for efConstruction in efConstruction_array:
                for maxConnections in maxConnections_array:
                
                    # import data
                    print("before import", benchmark_file, " curr: ", curr, " size: ", size)
                    import_time = import_into_weaviate(client, efConstruction, maxConnections, benchmark_file, curr, size)

                    # Find neighbors based on UUID and ef settings
                    results = []
                    for ef in ef_array:
                        result = conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections, size, multi)
                        result['importTime'] = import_time
                        results.append(result)
                    
                    # write json file
                    if not os.path.exists('results'):
                        os.makedirs('results')
                    output_json = 'results/weaviate_benchmark' + '__' + size_to_str(size) + '__' + str(efConstruction) + '__' + str(maxConnections) + '.json'
                    loguru.logger.info('Writing JSON file with results to: ' + output_json)
                    with open(output_json, 'w') as outfile:
                        json.dump(results, outfile)
        print("current size: ", client.query.aggregate("Benchmark").with_meta_count().do())
        curr, size = size, size + increment

    loguru.logger.info('completed')
