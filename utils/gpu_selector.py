import GPUtil

# # Function to select the GPU with the most available memory
# def get_best_gpu():
#     try:
#         best_gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=0.5, maxMemory=0.5)[0]
#         return best_gpu
#     except Exception as e:
#         print(f"Error selecting GPU: {e}")
#         return 0  # Default to GPU 0 if an error occurs

def get_best_gpu():
    """
    Selects the best GPU based on a combination of low usage load and free memory.

    It first attempts to find a GPU with a load less than 10%. If multiple such
    GPUs are found, it selects the one with the most free memory. If no GPU meets
    the low load criterion, it defaults to selecting the GPU with the most free
    memory among all available GPUs.

    Returns:
        int: The ID of the selected GPU. Defaults to 0 if an error occurs.
    """
    try:
        # Get a list of all available GPUs
        gpus = GPUtil.getGPUs()
        
        # Try to find a GPU with low load first (load < 10%)
        available_gpus = [gpu for gpu in gpus if gpu.load < 0.1]
        
        if available_gpus:
            # Sort by memory (descending) and pick the one with the most available memory
            available_gpus = sorted(available_gpus, key=lambda x: x.memoryFree, reverse=True)
            best_gpu = available_gpus[0]
        else:
            # If all GPUs are under load, select the one with the most free memory
            # Sort all GPUs by free memory in descending order
            gpus = sorted(gpus, key=lambda x: x.memoryFree, reverse=True)
            best_gpu = gpus[0]

        return best_gpu.id  # Return the GPU ID
        
    except Exception as e:
        print(f"Error selecting GPU: {e}")
        return 0  # Default to GPU 0 if an error occurs