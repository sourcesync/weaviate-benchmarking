from functions import *


if __name__ == '__main__':

    # 
    # Config
    #

    # This array should contain all the datasets you want to test.
    # Note that these files should be present in the "benchmark-data" directory
    # of this repo.  I recommend you only run one at a time, verify the results, 
    # and repeat.
    benchmark_file_array = [
#        ['deep-10K.hdf5','cosine'],
#        ['deep-1M.hdf5','cosine'],
#        ['deep-2M.hdf5','cosine'],
#        ['deep-5M.hdf5','cosine'],
        ['deep-10M.hdf5','cosine'],
#        ['deep-image-96-angular.hdf5', 'cosine'],
#        ['mnist-784-euclidean.hdf5', 'l2-squared'],
#        ['gist-960-euclidean.hdf5', 'l2-squared'],
#        ['glove-25-angular.hdf5', 'cosine']
    ]   
    
    # Set this to 'True' in order to time the non-HNSW building parts of the pipeline.
    skip_graph=False  

    # APU/Gemini config - set to 'None' for normal HNSW
    # gemini_parms = None
    gemini_parms = [    #{'nBits':768, 'searchType':'clusters'  },\
                        #{'nBits':512, 'searchType':'clusters' }, \
                        #{'nBits':256, 'searchType':'clusters' }, \
                        {'nBits':128, 'searchType':'clusters' }, \
                        {'nBits':64,  'searchType':'clusters' } ]
#    gemini_parms = [    {'nBits':768, 'searchType':'flat'  },\
#                        {'nBits':512, 'searchType':'flat' }, \
#                        {'nBits':256, 'searchType':'flat' }, \
#                        {'nBits':128, 'searchType':'flat' }, \
#                        {'nBits':64,  'searchType':'flat' } ]
    
    
    # Please don't change these unless you know what you are doing.
    if gemini_parms:
        CPUs = [1 ] #, 16, 32]
        efConstruction_array = None 
        maxConnections_array = None
        ef_array = None

    else:
        CPUs = [1, 16, 32]
        efConstruction_array = [64] #, 128]
        maxConnections_array = [16] #, 32]
        ef_array = [64, 128, 256, 512]

    # Change this to the same container/port you've configured in the docker-compose file
    if gemini_parms:
        #weaviate_url = 'http://weaviate-gsi:8084'
        weaviate_url = 'http://localhost:8084'
    else:
        #weaviate_url = 'http://weaviate:8084'
        weaviate_url = 'http://localhost:8091'
 
    # Starts the actual benchmark, prints "completed" when done
    print("Benchmark file array", benchmark_file_array)
    run_the_benchmarks(weaviate_url, CPUs, efConstruction_array, maxConnections_array, \
        ef_array, benchmark_file_array, skip_graph=skip_graph, gemini_parms=gemini_parms)
