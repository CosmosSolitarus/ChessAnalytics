import time
import cProfile
import pstats
from io import StringIO
import xgboost as xgb
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import csv
import os

class TimingStats:
    def __init__(self):
        self.combination_time = 0
        self.prediction_time = 0
        self.io_time = 0
        self.combinations_processed = 0
        self.start_time = time.time()

    def print_stats(self):
        total_time = time.time() - self.start_time
        print("\nTiming Statistics:")
        print(f"Total Runtime: {total_time:.2f} seconds")
        print(f"Combinations Processed: {self.combinations_processed:,}")
        print(f"Processing Rate: {self.combinations_processed/total_time:.2f} combinations/second")
        if self.combinations_processed > 0:
            print(f"\nPer Combination Averages:")
            print(f"Combination Generation: {(self.combination_time/self.combinations_processed)*1000:.3f} ms")
            print(f"XGBoost Prediction: {(self.prediction_time/self.combinations_processed)*1000:.3f} ms")
            print(f"I/O Operations: {(self.io_time/self.combinations_processed)*1000:.3f} ms")
        
        print(f"\nPercentage Breakdown:")
        print(f"Combination Generation: {(self.combination_time/total_time)*100:.1f}%")
        print(f"XGBoost Prediction: {(self.prediction_time/total_time)*100:.1f}%")
        print(f"I/O Operations: {(self.io_time/total_time)*100:.1f}%")

# Defines the bounds and precision of a parameter
# The parameter can be incremented by the step size from
# min to max to evaluate how an XGBoost's prediction changes
class LinearDatarange:
    def __init__(self, name, min_value, max_value, steps):
        self.name = name
        self.min = min_value
        self.max = max_value
        self.steps = steps      # Steps is assumed to be an integer >= 2
        self.step_size = (max_value - min_value) / (steps - 1)

# Defines the steps of a parameter
class NonlinearDatarange:
    def __init__(self, name, step_array):
        self.name = name
        self.step_array = step_array    # Steps is assumed to be an array
        self.steps = len(step_array)

# Defines the relation between one-hot encoded categorical variables
class OneHotDatarange:
    def __init__(self, names):
        self.names = names
        self.steps = len(names)

# A collection of dataranges. Represent the entire search space 
# for a set of parameters in an XGBoost model
class Dataspace:
    def __init__(self):
        self.dataranges = []
        self.size = 1

    # Appends a datarange to the ranges list and updates size
    def append(self, datarange):
        self.dataranges.append(datarange)
        self.size *= datarange.steps

    # Convert a given index to a specific combination of values
    def index_to_combination(self, index):
        num_combinations = self.size
        combination = []

        if index < 0 or index > num_combinations - 1:
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {num_combinations-1}.")

        for datarange in self.dataranges:
            steps = datarange.steps
            
            if isinstance(datarange, LinearDatarange):
                combination.append(datarange.min + round((index % steps) * datarange.step_size, 3))
            elif isinstance(datarange, NonlinearDatarange):
                combination.append(datarange.step_array[index % steps])
            elif isinstance(datarange, OneHotDatarange):
                current_index = steps - (index % steps) - 1

                for i in range(steps):  # Add a 0 for all indices except the current one
                    combination.append(1 if i == current_index else 0)
            
            index //= steps

        return combination

    # Divides the total combinations evenly among a number of workers
    # The last worker is assigned remainder combinations, with a worst
    # case of num_workers - 1 extra combinations to check (trivial)
    def get_worker_indices(self, num_workers):
        num_combinations = self.size

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

# Determines whether the model's columns match the dataspace's
# columns, and there are no extras for either
def validate_columns(model, dataspace):
    # Get column names
    model_columns = model.feature_names
    dataspace_columns = []

    for datarange in dataspace.dataranges:
        if isinstance(datarange, LinearDatarange):
            dataspace_columns.append(datarange.name)
        elif isinstance(datarange, NonlinearDatarange):
            dataspace_columns.append(datarange.name)
        elif isinstance(datarange, OneHotDatarange):
            dataspace_columns.extend(datarange.names)

    # Check for mismatches
    extra_model_columns = set(model_columns) - set(dataspace_columns)
    extra_dataspace_columns = set(dataspace_columns) - set(model_columns)

    if extra_model_columns or extra_dataspace_columns:
        raise ValueError(
            f"Column mismatch detected!\n"
            f"Extra in model: {extra_model_columns}\n"
            f"Extra in dataspace: {extra_dataspace_columns}"
        )

    print("Column validation passed.")

# Worker function to evaluate combinations
def worker_search(model, dataspace, start_index, num_combinations, worker_id):
    stats = TimingStats()
    feature_names = model.feature_names
    worker_file = f"worker_{worker_id}.csv"
    
    # Set up for batched I/O
    PREDICTION_BATCH_SIZE = 100000      # Up from 10000
    IO_BATCH_SIZE = 100000              # Up from 100000
    io_buffer = []
    
    # Test prediction setup and determine if classification
    t0 = time.time()
    test_input = dataspace.index_to_combination(start_index)
    stats.combination_time += time.time() - t0
    
    t0 = time.time()
    test_matrix = xgb.DMatrix([test_input], feature_names=feature_names)
    test_pred = model.predict(test_matrix, output_margin=False)
    stats.prediction_time += time.time() - t0
    
    is_classification = len(test_pred.shape) > 1
    
    # Prepare headers
    if is_classification:
        num_classes = test_pred.shape[1]
        headers = [f'prediction_class_{i}' for i in range(num_classes)]
    else:
        headers = ['prediction']
    
    feature_idx = 0
    for datarange in dataspace.dataranges:
        if isinstance(datarange, LinearDatarange):
            headers.append(datarange.name)
            feature_idx += 1
        elif isinstance(datarange, NonlinearDatarange):
            headers.append(datarange.name)
            feature_idx += 1
        elif isinstance(datarange, OneHotDatarange):
            headers.extend(datarange.names)
            feature_idx += len(datarange.names)

    # Initialize CSV file with headers
    with open(worker_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    def write_buffer_to_csv():
        if io_buffer:
            t0 = time.time()
            with open(worker_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(io_buffer)
            stats.io_time += time.time() - t0
            io_buffer.clear()
    
    # Process combinations in batches
    for batch_start in range(start_index, start_index + num_combinations, PREDICTION_BATCH_SIZE):
        batch_end = min(batch_start + PREDICTION_BATCH_SIZE, start_index + num_combinations)
        batch_size = batch_end - batch_start
        batch_combinations = []
        batch_inputs = []
        
        # Generate combinations
        for idx in range(batch_start, batch_end):
            t0 = time.time()
            combination = dataspace.index_to_combination(idx)
            stats.combination_time += time.time() - t0
            batch_combinations.append(combination)
            
            inputs = {}
            feature_idx = 0
            for datarange in dataspace.dataranges:
                if isinstance(datarange, LinearDatarange):
                    inputs[datarange.name] = combination[feature_idx]
                    feature_idx += 1
                elif isinstance(datarange, NonlinearDatarange):
                    inputs[datarange.name] = combination[feature_idx]
                    feature_idx += 1
                elif isinstance(datarange, OneHotDatarange):
                    for name in datarange.names:
                        inputs[name] = combination[feature_idx]
                        feature_idx += 1
            
            sorted_inputs = [inputs[name] for name in feature_names]
            batch_inputs.append(sorted_inputs)
        
        # Make predictions
        t0 = time.time()
        input_matrix = xgb.DMatrix(batch_inputs, feature_names=feature_names)
        predictions = model.predict(input_matrix, output_margin=False)
        stats.prediction_time += time.time() - t0
        
        # Prepare rows for buffer
        for idx, (combination, prediction) in enumerate(zip(batch_combinations, predictions)):
            row = []
            if is_classification:
                row.extend(prediction)
            else:
                row.append(prediction)
            
            feature_idx = 0
            for datarange in dataspace.dataranges:
                if isinstance(datarange, LinearDatarange):
                    row.append(combination[feature_idx])
                    feature_idx += 1
                elif isinstance(datarange, NonlinearDatarange):
                    row.append(combination[feature_idx])
                    feature_idx += 1
                elif isinstance(datarange, OneHotDatarange):
                    for _ in datarange.names:
                        row.append(combination[feature_idx])
                        feature_idx += 1
            
            io_buffer.append(row)
            
            # Write buffer if it reaches the size limit
            if len(io_buffer) >= IO_BATCH_SIZE:
                write_buffer_to_csv()
        
        stats.combinations_processed += batch_size
    
    # Write any remaining rows
    write_buffer_to_csv()
    stats.print_stats()
    return worker_file

# Provided an XGBoost model and information about its columns, search
# all possible combinations of inputs and save the outputs to CSVs
def dataspace_search(model, dataspace, num_threads=max(1, cpu_count() - 1)):    
    # Enable profiling
    pr = cProfile.Profile()
    pr.enable()
    
    print(f"Starting search with {dataspace.size:,} combinations")
    print(f"Using {num_threads} threads ({dataspace.size/num_threads:,.0f} combinations per thread)")
    print(f"Estimated time: {dataspace.size * 0.019 / (60 * 1000):,.0f} minutes.")
    
    validate_columns(model, dataspace)
    
    worker_indices = dataspace.get_worker_indices(num_threads)
    worker_files = []
    output_file = "Dataspace.csv"

    def process_worker_range(args):
        worker_id, (start_index, num_combinations) = args
        return worker_search(model, dataspace, start_index, num_combinations, worker_id)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        worker_args = enumerate([
            (start_index, num_combinations)
            for start_index, num_combinations in worker_indices
        ])
        worker_files = list(executor.map(process_worker_range, worker_args))
    parallel_time = time.time() - t0

    print(f"\nParallel processing time: {parallel_time:.2f} seconds")
    
    # Combine results efficiently using file concatenation
    t0 = time.time()
    with open(worker_files[0], 'r', newline='') as first_file:
        header = next(csv.reader(first_file))
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        # Use larger buffer size for combining files
        buffer_size = 1024 * 1024  # 1MB buffer
        for worker_file in worker_files:
            with open(worker_file, 'r', newline='') as infile:
                # Skip header
                next(infile)
                while True:
                    chunk = infile.read(buffer_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
            os.remove(worker_file)
    
    combine_time = time.time() - t0
    print(f"File combination time: {combine_time:.2f} seconds")
    
    # Disable profiling and print results
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

def test1():
    # Load an XGBoost model from a saved JSON file
    model = xgb.Booster()
    model.load_model("json/model.json")

    # Create dataspace
    ds = Dataspace()

    # Add one-hot encoded columns for days of the week (Sunday through Friday)
    ds.append(OneHotDatarange(['IsSunday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday']))

    # Add binary columns for result-related features
    ds.append(LinearDatarange('LastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('LastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('LastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1

    # Add continuous columns with defined ranges and steps
    ds.append(LinearDatarange('TimeOfDay', 0, 82800, 24))
    ds.append(LinearDatarange('GameOfDay', 1, 10, 10))
    ds.append(LinearDatarange('GameOfWeek', 10, 100, 10))
    ds.append(LinearDatarange('EloDifference', -100, 100, 9))
    ds.append(NonlinearDatarange('TimeSinceLast', [5*60, 15*60, 3600, 2*3600, 4*3600, 18*3600, 36*3600, 72*3600]))

    # Run MinMaxSearch
    dataspace_search(model, ds, num_threads=7)

def test2():
    model = xgb.Booster()
    model.load_model("json/model.json")

    print("Model feature names:", model.feature_names)

    # Assuming you have the model loaded
    model = xgb.Booster()
    model.load_model("json/model.json")

    # The feature names expected by the model
    feature_names = ['IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday', 'IsSunday', 
                    'TimeOfDay', 'GameOfDay', 'GameOfWeek', 'EloDifference', 'LastResultIsWin', 'LastResultIsDraw', 
                    'LastResultIsLoss', '2ndLastResultIsWin', '2ndLastResultIsDraw', '2ndLastResultIsLoss', 'TimeSinceLast']

    # Example data (must be aligned with the feature names)
    input_data = np.array([[1, 0, 0, 0, 0, 0, 0, 36000, 5, 30, 10, 1, 0, 0, 1, 0, 0, 3600]])

    # Create DMatrix and set feature names explicitly
    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)

    # Make prediction
    prediction = model.predict(dmatrix)

    print("Prediction:", prediction)

# passes
def test3():
    ds = Dataspace()

    ds.append(LinearDatarange('Location', 100, 500, 5))
    ds.append(NonlinearDatarange('Times', [1,2,4,8,16]))
    ds.append(OneHotDatarange(['Red', 'Green', 'Blue']))

    print(ds.size)

    for i in range(ds.size):
        print(ds.index_to_combination(i))

    print(ds.get_worker_indices(5))

    print(ds.get_worker_indices(10))

def test4():
    start = time.time() * 1000.0
    num_threads = 7
    
    # Load an XGBoost model from a saved JSON file
    model = xgb.Booster()
    model.load_model("json/model.json")

    # Create dataspace
    ds = Dataspace()

    # Add one-hot encoded columns for days of the week (Sunday through Friday)
    ds.append(OneHotDatarange(['IsSunday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday']))

    # Add binary columns for result-related features
    ds.append(LinearDatarange('LastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('LastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('LastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    ds.append(LinearDatarange('2ndLastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1

    # Add continuous columns with defined ranges and steps
    ds.append(LinearDatarange('TimeOfDay', 0, 82800, 24))
    ds.append(LinearDatarange('GameOfDay', 1, 10, 10))
    ds.append(LinearDatarange('GameOfWeek', 10, 100, 10))
    ds.append(LinearDatarange('EloDifference', -100, 100, 9))
    ds.append(NonlinearDatarange('TimeSinceLast', [5*60, 15*60, 3600, 2*3600, 4*3600, 18*3600, 36*3600, 72*3600]))

    # Run MinMaxSearch
    dataspace_search(model, ds, num_threads=num_threads)

    end = time.time() * 1000.0
    real_time = (end-start) / 1000

    print(f"Total real time: {real_time} seconds.")
    print(f"Total core time: {num_threads * real_time} seconds.")
    print(f"Average real time per 1000 combinations: {1000 * real_time / ds.size} seconds.")
    print(f"Average core time per 1000 combinations: {num_threads * 1000 * real_time / ds.size} seconds.")

def main():
    #test1()
    #test2()
    #test3()
    test4()

if __name__ == "__main__":
    main()