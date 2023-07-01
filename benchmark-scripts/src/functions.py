import os
import uuid
import json
import time
import datetime
import subprocess
import h5py
import weaviate
import loguru
import sys

import gemini

#
# Config
#

# You don't want a vector cache if you are benchmarking the vector search algorithm.
VectorCacheMaxObjects = 0

# Gemini index config ( 'nBits' and 'searchType should be the ones you really care about 
GEMINI_PARAMETERS   = {'skip': False, 'searchType': 'flat', 'centroidsHammingK': 5000, 'centroidsRerank': 4000, 'hammingK': 3200, 'nBits': -1 }

def add_batch(client, c, vector_len, skip_graph=False, verbose=True):
    '''Adds batch to Weaviate and returns
       the time it took to complete in seconds.'''
    
    if skip_graph:
        print("Skipping add_batch - " + str(c) + ' of ' +str(vector_len) + " objects")
        return 0

    start_time = datetime.datetime.now()
    results = client.batch.create_objects()
    stop_time = datetime.datetime.now()
    handle_results(results)
    run_time = stop_time - start_time
    if (c % 10000) == 0:
        loguru.logger.info('Import status => added ' + str(c) + ' of ' + str(vector_len) + ' objects')

    if verbose: print("add_batch stats", run_time.total_seconds(), run_time.seconds)

    return run_time.total_seconds()


def handle_results(results):
    '''Handle error message from batch requests
       logs the message as an info message.'''
    if results is not None:
        for result in results:
            if 'result' in result and 'errors' in result['result'] and  'error' in result['result']['errors']:
                for message in result['result']['errors']['error']:
                    loguru.logger.error(message['message'])


def match_results(test_set, weaviate_result_set, k):
    '''Match the reults from Weaviate to the benchmark data.
       If a result is in the returned set, score goes +1.
       Because there is checked for 100 neighbors a score
       of 100 == perfect'''

    # set score
    score = 0

    # return if no result
    if weaviate_result_set['data']['Get']['Benchmark'] == None:
        return score

    #print("match0", test_set[:k], weaviate_result_set['data']['Get']['Benchmark'])

    # create array from Weaviate result
    weaviate_result_array = []
    for weaviate_result in weaviate_result_set['data']['Get']['Benchmark'][:k]:
        weaviate_result_array.append(weaviate_result['counter'])

    # match scores
    #print("match", test_set[:k], weaviate_result_array)
    for nn in test_set[:k]:
        if nn in weaviate_result_array:
            score += 1
    
    return score


def run_speed_test(l, CPUs,weaviate_url):
    '''Runs the actual speed test in Go'''
    process = subprocess.Popen(['./benchmarker','dataset', '-u', weaviate_url, '-c', 'Benchmark', '-q', 'queries.json', '-p', str(CPUs), '-f', 'json', '-l', str(l)], stdout=subprocess.PIPE)
    result_raw = process.communicate()[0].decode('utf-8')
    return json.loads(result_raw)


def conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections, gemini_parm=False):
    '''Conducts the benchmark, note that the NN results
       and speed test run seperatly from each other'''

    # result obj
    if gemini_parm:
        results = {
            'benchmarkFile': benchmark_file[0],
            'distanceMetric': benchmark_file[1],
            'totalTested': 0,
            'nBits': gemini_parm['nBits'],
            'searchType': gemini_parm['searchType'],
            'recall': {
                '10': {
                    'highest': 0,
                    'lowest': 100,
                    'average': 0
                },
            },
        'requestTimes': {}
        }
    else:
        results = {
            'benchmarkFile': benchmark_file[0],
            'distanceMetric': benchmark_file[1],
            'totalTested': 0,
            'ef': ef,
            'efConstruction': efConstruction,
            'maxConnections': maxConnections,
            'recall': {
                '10': {
                    'highest': 0,
                    'lowest': 100,
                    'average': 0
                },
            },
            'requestTimes': {}
        }

    # update schema for ef setting
    loguru.logger.info('Update "ef" to ' + str(ef) + ' in schema')

    if not gemini_parm:
        print("Updating schema with ef", ef ) 
        client.schema.update_config('Benchmark', { 'vectorIndexConfig': { 'ef': ef } })

    #
    # Run the score test
    #
    c = 0
    all_scores = {
            '10':[],
        }

    loguru.logger.info('Find neighbors with ef = ' + str(ef))
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        test_vectors = f['test']
        test_vectors_len = len(f['test'])
        for test_vector in test_vectors:

            # set certainty for  l2-squared
            nearVector = { "vector": test_vector.tolist() }
            
            # Start request
            if gemini_parm:
                query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(10).do()    
            else:
                query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(100).do()    

            for k in [10]: 
                k_label=f'{k}'
                score = match_results(f['neighbors'][c], query_result, k)
                if score == 0:
                    loguru.logger.info('There is a 0 score, this most likely means there is an issue with the dataset OR you have very low index settings. Found for vector: ' + str(test_vector[0]))
                all_scores[k_label].append(score)
                
                # set if high and low score
                if score > results['recall'][k_label]['highest']:
                    results['recall'][k_label]['highest'] = score
                if score < results['recall'][k_label]['lowest']:
                    results['recall'][k_label]['lowest'] = score

            # log ouput
            if (c % 1000) == 0:
                loguru.logger.info('Validated ' + str(c) + ' of ' + str(test_vectors_len))

            c+=1

    #
    # Run the speed test
    #
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
        results['requestTimes']['limit_10'] = run_speed_test(10, CPUs, weaviate_url) #if False else -1

    # add final results
    results['totalTested'] = c
    results['totalDatasetSize'] = train_vectors_len
    for k in [ '10']: 
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


def import_into_weaviate(client, efConstruction, maxConnections, benchmark_file, skip_graph=False, gemini_parm=False):
    '''Imports the data into Weaviate'''
   
    if gemini_parm:
        import os
        allocid = os.getenv("GEMINI_ALLOCATION_ID")
        gemini.benchmark_prepare( allocid )
 
    # variables
    if gemini_parm:
        benchmark_import_batch_size = 1 # TODO: We need to support true batching!
    else:
        benchmark_import_batch_size = 10000

    benchmark_class = 'Benchmark'
    import_time = 0.0

    if skip_graph:
        print("Skipping schema query...")
    else:
        # Delete schema if available
        current_schema = client.schema.get()
        if len(current_schema['classes']) > 0:
            print("Removing previous classes...")
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
                "vectorCacheMaxObjects": VectorCacheMaxObjects,
                "distance": benchmark_file[1]
            }
        }]
    }

    if skip_graph:
        print("Skipping schema create...")
    else:
        if gemini_parm:
            for ky in gemini_parm.keys():
                if ky not in GEMINI_PARAMETERS.keys():
                    raise Exception("Invalid Gemini parm=", ky)
                else:
                    print("Setting Gemini parm", ky, "to", gemini_parm[ky])
                GEMINI_PARAMETERS[ky] = gemini_parm[ky]
            schema['classes'][0]['vectorIndexConfig'] = GEMINI_PARAMETERS
            schema['classes'][0]['vectorIndexType'] = "gemini"
            print("Setting APU/Gemini index config", schema['classes'][0]['vectorIndexConfig'])
        print("Setting schema...", schema)
        client.schema.create(schema)
        print("Scheme set. Confirming...")
        current_schema = client.schema.get()
        print("Current schema is", current_schema)



    # Import
    if gemini_parm:
        loguru.logger.info('Start import process for ' + benchmark_file[0] + ', parms=' + str(gemini_parm))
    else:
        loguru.logger.info('Start import process for ' + benchmark_file[0] + ', ef' + str(efConstruction) + ', maxConnections' + str(maxConnections))

    import os
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        vectors = f['train']
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
                import_time += add_batch(client, c, vector_len, skip_graph, benchmark_import_batch_size>1)
                batch_c = 0
            if (c%1000)==0: 
                print("Processed %d/%d vectors, add batch import time thus far=%f" % (c+1, vector_len,import_time ))
            c += 1
            batch_c += 1
        import_time += add_batch(client, c, vector_len, skip_graph, benchmark_import_batch_size>1)
    loguru.logger.info('done importing ' + str(c) + ' objects in ' + str(import_time) + ' seconds')

    return import_time

def parse_gemini_result(result):
    '''Parse a query result into something actionable.'''

    async_try_again = False
    errors = []
    data = None

    # First loop through errors if any.  
    # We look for "gemini async build" messages 
    # and don't interpret them as errors.
    if "errors" in result.keys():
        errs = result["errors"]
        for err in errs:
            if "message" in err.keys():
                mesg = err["message"]
                if mesg.find("vector search: Async index build is in progress.")>=0:
                    async_try_again = True
                elif mesg.find("vector search: Async index build completed.")>=0:
                    async_try_again = True
                else:
                    errors.append(err)

    elif "data" in result.keys():
        data = result["data"]

    return async_try_again, errors, data

def gemini_wait(client, benchmark_file):

    print("Post import search for gemini status...")

    start_time = datetime.datetime.now()

    nearVector = None
    with h5py.File('/var/hdf5/' + benchmark_file[0], 'r') as f:
        test_vectors = f['test']
        test_vectors_len = len(f['test'])
        test_vector = test_vectors[0]
        nearVector = { "vector": test_vector.tolist() }

    if not nearVector:
        print("test vector is invalid")
        return -1

    # loop wait is here
    consec_errs = 0
    fail = False
    while True:

        query_result = client.query.get("Benchmark", ["counter"]).with_near_vector(nearVector).with_limit(10).do()

        # Interpret the results
        async_try_again, errors, data = parse_gemini_result(query_result)
        if async_try_again:
            print("Gemini is asynchronously building an index, and has asked us to try the search again a little later...")
            time.sleep(2)
            continue
        elif errors:
            print("We got search errors->", errors)
            consec_errs += 1
            if consec_errs > 5:
                print("Too many errors.  Let's stop here.")
                fail = True
                break
        elif data:
            print("Successful search, data->", data)
            consec_errs = 0
            break
        else:
            print("Unknown result! Let's stop here.")
            fail = True
            break

    stop_time = datetime.datetime.now()
    gemini_train_time = stop_time - start_time
    
    print("Post import search gemini wait done.")

    if fail:
        return -1
    else:
        print("gemini wait stats - ", gemini_train_time.total_seconds(), gemini_train_time.seconds)
        return gemini_train_time.total_seconds()

def run_the_benchmarks(weaviate_url, CPUs, efConstruction_array, maxConnections_array, ef_array, benchmark_file_array, skip_graph=False, gemini_parms=False):
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

    if gemini_parms:

        for benchmark_file in benchmark_file_array:

            for gemini_parm in gemini_parms:

                start_time = datetime.datetime.now()
                import_time = import_into_weaviate(client, None, None, benchmark_file, skip_graph=skip_graph, gemini_parm=gemini_parm)
                stop_time = datetime.datetime.now()
                wall_time_import_time = stop_time - start_time

                # for APU/gemini, we need to initiate training and wait for it to finish
                train_time = gemini_wait(client, benchmark_file)
                if (train_time<0):
                    print("Gemini wait failed.")

                for CPU in CPUs:

                    # Find neighbors based on UUID and ef settings
                    results = []
                    if skip_graph:
                        result = {}
                        print("Skipping the actual benchmark...")
                    else:
                        result = conduct_benchmark(weaviate_url, CPU, None, client, benchmark_file, None, None, gemini_parm=gemini_parm)
                    result['importTime'] = import_time
                    result['startTime'] = start_time.timestamp()
                    result['wallImportTime'] = wall_time_import_time.total_seconds()
                    result['vectorCacheMaxObjects'] = VectorCacheMaxObjects 
                    result['trainTime'] = train_time
                    results.append(result)
                    print("Dumping results->", results)
             
                    # write json file
                    if not os.path.exists('results'):
                        os.makedirs('results')
                    if skip_graph:
                        output_json = 'results/weaviate_benchmark' + '__' + \
                            benchmark_file[0] + '__skip_graph__gemini.json'
                    else:
                        output_json = 'results/weaviate_benchmark' + '__' + \
                            benchmark_file[0] + '__' + "gemini" + '__' + \
                                str(gemini_parm['nBits']) + '__' + str(gemini_parm['searchType']) + '__' + str(CPU) + '.json'

                    loguru.logger.info('Writing JSON file with results to: ' + output_json)
                    with open(output_json, 'w') as outfile:
                        json.dump(results, outfile)

    else: # regular hnsw
        # iterate over settings
        for benchmark_file in benchmark_file_array:
            for efConstruction in efConstruction_array:
                for maxConnections in maxConnections_array:
                   
                    # import data
                    start_time = datetime.datetime.now()
                    import_time = import_into_weaviate(client, efConstruction, maxConnections, benchmark_file, skip_graph=skip_graph)
                    stop_time = datetime.datetime.now()
                    wall_time_import_time = stop_time - start_time

                    for CPU in CPUs:

                        # Find neighbors based on UUID and ef settings
                        results = []
                        for ef in ef_array:
                            if skip_graph:
                                result = {}
                                print("Skipping benchmark...")
                            else:
                                result = conduct_benchmark(weaviate_url, CPUs, ef, client, benchmark_file, efConstruction, maxConnections)
                            result['importTime'] = import_time
                            result['startTime'] = start_time.timestamp()
                            result['wallImportTime'] = wall_time_import_time.total_seconds()
                            result['vectorCacheMaxObjects'] = VectorCacheMaxObjects 
                            result['trainTime'] = -1
                            results.append(result)
                            if skip_graph: break
                        
                        # write json file
                        if not os.path.exists('results'):
                            os.makedirs('results')
                        if skip_graph:
                            output_json = 'results/weaviate_benchmark' + '__' + benchmark_file[0] + '__skip_graph__' + str(CPU) + '.json'
                        else:
                            output_json = 'results/weaviate_benchmark' + '__' + \
                                benchmark_file[0] + '__' + str(efConstruction) + '__' + str(maxConnections) + '__' + str(CPU) + '.json'
                        loguru.logger.info('Writing JSON file with results to: ' + output_json)
                        with open(output_json, 'w') as outfile:
                            json.dump(results, outfile)

    loguru.logger.info('completed')
