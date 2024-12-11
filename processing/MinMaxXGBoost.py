import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

class Datarange:
    def __init__(self, name, min_value, max_value, steps):
        self.name = name
        self.min = min_value
        self.max = max_value
        self.steps = steps      # Steps is assumed to be an integer >= 2
        self.step_size = (max_value - min_value) / (steps - 1)
        self.type = 'numeric'

class OneHotDatarange:
    def __init__(self, names):
        self.names = names
        self.type = 'onehot'
        self.steps = len(names)

class Dataspace:
    def __init__(self):
        self.ranges = []

    # Appends a datarange to the ranges list.
    def append(self, datarange):
        self.ranges.append(datarange)

    # Calculates the total number of combinations across all ranges.
    def size(self):
        total_size = 1

        for range_obj in self.ranges:
            total_size *= range_obj.steps
        
        return total_size

    # Convert a given index to a specific combination of values.
    def index_to_combination(self, index):
        num_combinations = self.size()
        combination = []

        if index < 0 or index > num_combinations - 1:
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {num_combinations-1}.")

        for range_obj in self.ranges:
            steps = range_obj.steps
            
            if range_obj.type == 'numeric':
                value = range_obj.min + round((index % steps) * range_obj.step_size, 3)
                combination.append(value)
            else:  # 'onehot'
                current_index = steps - (index % steps) - 1

                for i in range(steps):  # Add a 0 for all indices except the current one
                    combination.append(1 if i == current_index else 0)
            
            index //= steps

        return combination

    # Divides the total combinations evenly among a number of workers.
    # The last worker is assigned remainder combinations, with a worst
    # case of num_workers - 1 extra combinations to check (trivial).
    def get_worker_indices(self, num_workers):
        num_combinations = self.size()

        combinations_per_worker = num_combinations // num_workers
        leftover_combinations = num_combinations % num_workers

        worker_indices = []
        start_index = 0

        for i in range(num_workers):
            if i != num_workers - 1:
                num_combinations_worker = combinations_per_worker
            else:
                num_combinations_worker = combinations_per_worker + leftover_combinations

            worker_indices.append((start_index, num_combinations_worker))
            start_index += num_combinations_worker

        return worker_indices

def validate_columns(model, dataspace):
    # Get column names
    model_columns = model.feature_names
    dataspace_columns = []

    for range_obj in dataspace.ranges:
        if range_obj.type == 'numeric':
            dataspace_columns.append(range_obj.name)
        elif range_obj.type == 'onehot':
            dataspace_columns.extend(range_obj.names)

    # Check for mismatches
    extra_model_columns = set(model_columns) - set(dataspace_columns)
    extra_dataspace_columns = set(dataspace_columns) - set(model_columns)

    if extra_model_columns or extra_dataspace_columns:
        raise ValueError(
            f"Column mismatch detected!\n"
            f"Extra in model: {extra_model_columns}\n"
            f"Extra in dataspace: {extra_dataspace_columns}"
        )

def MinMaxSearch(model, dataspace, top_n=1, num_threads=None):
    if num_threads is None:
        num_threads = max(1, cpu_count() - 1)

    # Step 1: Warn user about the total combinations
    total_combinations = dataspace.size()
    print(f"Warning: The total number of combinations to search is {total_combinations}.")
    print(f"Using {num_threads} threads for parallel processing. ({total_combinations / num_threads} combinations per thread.)")

    # Step 2: Validate model and dataspace compatibility
    validate_columns(model, dataspace)

    # Step 3: Parallelized Search
    # This is where the worker threads will be launched to perform searches.
    # Each worker will receive a range of indices based on dataspace.get_worker_indices().

    # Step 4: Combine Results
    # After parallel execution, combine the min/max values and top_n results
    # from each worker into a single JSON-serializable structure.

    # Step 5: Output Results to JSON
    results = {
        "min": [],  # Placeholder for minimum values and their inputs
        "max": [],  # Placeholder for maximum values and their inputs
    }

    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}.")