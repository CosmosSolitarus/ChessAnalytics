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

def test():
    # Create a new Dataspace
    dataspace = Dataspace()

    # Add Dataranges and OneHotDataranges
    dataspace.append(Datarange("Binary", 0, 1, 2))
    #dataspace.append(Datarange("Ordinal", 0, 9, 10))
    dataspace.append(Datarange("Continuous", 0, 1, 3))
    dataspace.append(OneHotDatarange(["Red", "Green", "Blue"]))

    # Test size
    num_combinations = dataspace.size()
    print(f"Total Combinations: {num_combinations}")

    for i in range(num_combinations):
        print(f"{i}: {dataspace.index_to_combination(i)}")

    # # Test worker indices
    # num_workers = 7
    # worker_indices = dataspace.get_worker_indices(num_workers)
    # print(f"Worker Indices: {worker_indices}")

    # # Test index to combination
    # for worker_index in worker_indices:
    #     starting_combination = dataspace.index_to_combination(worker_index[0])
    #     print(f"{worker_index}: {starting_combination}")

def main():
    test()

if __name__ == "__main__":
    main()