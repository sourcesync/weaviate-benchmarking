import os
import socket

import swagger_client
from swagger_client.models import *




def benchmark_prepare(allocation, unload=True, wipe=False):
    '''Prepares an APU/Gemini system for benchmarking which includes unloading any datasets.'''

    if unload:
        # Setup connection to local FVS api
        server = socket.gethostbyname(socket.gethostname())
        port = "7761"
        version = 'v1.0'

        # Create FVS api objects
        config = swagger_client.configuration.Configuration()
        api_config = swagger_client.ApiClient(config)
        gsi_boards_apis = swagger_client.BoardsApi(api_config)
        gsi_datasets_apis = swagger_client.DatasetsApi(api_config)

        # Configure the FVS api
        config.verify_ssl = False
        config.host = f'http://{server}:{port}/{version}'

        # Capture the supplied allocation id
        Allocation_id = allocation

        # Set default header
        api_config.default_headers["allocationToken"] = Allocation_id

        # Print dataset count
        print("Gemini: Getting total datasets...")
        dsets = gsi_datasets_apis.controllers_dataset_controller_get_datasets_list(allocation_token=Allocation_id)
        print(f"Gemini: Number of datasets:{len(dsets.datasets_list)}")

        # if no datasets skip everything
        if len(dsets.datasets_list) > 0:
            # Print loaded dataset count
            print("Gemini: Getting loaded datasets for allocation token: ", Allocation_id)
            loaded = gsi_boards_apis.controllers_boards_controller_get_allocations_list(Allocation_id)
            print(f"Gemini: Number of loaded datasets: {len(loaded.allocations_list[Allocation_id]['loadedDatasets'])}")
            # check loaded dataset count
            if len(loaded.allocations_list[Allocation_id]["loadedDatasets"]) > 0:
                # Unloading all datasets
                print("Gemini: Unloading all loaded datasets...")
                loaded = loaded.allocations_list[Allocation_id]["loadedDatasets"]
                for data in loaded:
                    dataset_id = data['datasetId']
                    resp = gsi_datasets_apis.controllers_dataset_controller_unload_dataset(
                                UnloadDatasetRequest(allocation_id=Allocation_id, dataset_id=dataset_id),
                                allocation_token=Allocation_id)
                    if resp.status != 'ok':
                        print(f"Gemini: error unloading dataset: {dataset_id}")

                # Getting current number of loaded datasets
                curr = gsi_boards_apis.controllers_boards_controller_get_allocations_list(Allocation_id)
                print(f"Gemini: Unloaded datasets, current loaded dataset count: {len(curr.allocations_list[Allocation_id]['loadedDatasets'])}")

        # Full wipe: delete all datasets
        if wipe == True:
            wipe = input("are you super sure? y/[n]: ")
            if wipe == "y":
                print("removing all datasets...")
                for data in dsets.datasets_list:
                    dataset_id = data['id']
                    resp = gsi_datasets_apis.controllers_dataset_controller_remove_dataset(\
                            dataset_id=dataset_id, allocation_token=Allocation_id)
                    if resp.status != "ok":
                        print(f"Error removing dataset: {dataset_id}")

        else:
            print("Gemini: Currently no loaded datasets. Done.")
    else:
        print("Gemini: Done")

