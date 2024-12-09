class Dataspace:
    def __init__(self, name, min_value, max_value, steps):
        self.name = name
        self.min = min_value
        self.max = max_value
        self.steps = steps      # Steps is assumed to be an integer >= 2
        self.step_size = (max_value - min_value) / (steps - 1)

class OneHotDataspace:
    def __init__(self, names):
        self.names = names

# calculates the total number of combinations from given dataspaces
def size(dataspaces):
    total_size = 1

    for dataspace in dataspaces:
        total_size *= dataspace.steps
    
    return total_size

# given the dataspaces and number of workers, determines that
# starting index and number of combinations to search for each
# worker by evenly dividing the total number of combinations
def get_worker_indices(dataspaces, num_workers):
    num_combinations = size(dataspaces)

    combinations_per_worker = num_combinations // num_workers
    leftover_combinations = num_combinations % num_workers

    worker_indices = []
    start_index = 0

    # assign the last worker any leftover combinations that can't be evenly
    # divided by the number of workers. Worst case is num_workers - 1
    # extra combinations to check, which is trivial
    for i in range(num_workers):
        if i != num_workers - 1:
            num_combinations = combinations_per_worker
        else:
            num_combinations = combinations_per_worker + leftover_combinations

        worker_indices.append((start_index, num_combinations))
        start_index += num_combinations

    return worker_indices

# given the dataspaces and an index, returns the combination 
# corresponding to that index
def index_to_combination(dataspaces, index):
    num_combinations = size(dataspaces)
    combination = [dataspace.min for dataspace in dataspaces]

    if index < 0 or index > num_combinations - 1:
        raise ValueError(f"Invalid index: {index}. Must be between 0 and {num_combinations-1}.")

    num_dataspaces = len(dataspaces)

    for i in range(num_dataspaces -1, -1, -1):
        steps = dataspaces[i].steps
        combination[i] += round((index % steps) * dataspaces[i].step_size, 3)
        index //= steps

    return combination

def test():
    dataspaces = [
        Dataspace("Binary", 5, 7, 2),
        Dataspace("Ordinal", 2, 20, 10),
        Dataspace("Continuous", -4, -2, 3)
    ]

    num_combinations = size(dataspaces)
    print(f"Total Combinations: {num_combinations}")

    num_workers = 7
    worker_indices = get_worker_indices(dataspaces, num_workers)
    print(f"Worker Indices: {worker_indices}")

    for worker_index in worker_indices:
        starting_combination = index_to_combination(dataspaces, worker_index[0])
        print(f"{worker_index}: {starting_combination}")

def main():
    test()

if __name__ == "__main__":
    main()