from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import xgboost as xgb
import numpy as np
import os, csv

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
    feature_names = model.feature_names
    worker_file = f"worker_{worker_id}.csv"
    
    # Determine if model is classification or regression by making a test prediction
    test_input = dataspace.index_to_combination(start_index)
    test_matrix = xgb.DMatrix([test_input], feature_names=feature_names)
    test_pred = model.predict(test_matrix, output_margin=False)
    is_classification = len(test_pred.shape) > 1
    
    # Prepare CSV headers
    if is_classification:
        num_classes = test_pred.shape[1]
        headers = [f'prediction_class_{i}' for i in range(num_classes)]
    else:
        headers = ['prediction']
    
    # Add feature names to headers
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

    # Write results directly to CSV
    with open(worker_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for idx in range(start_index, start_index + num_combinations):
            combination = dataspace.index_to_combination(idx)
            
            # Convert combination into a dictionary format
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

            # Sort inputs by feature names to match model's expected order
            sorted_inputs = [inputs[name] for name in feature_names]
            input_matrix = xgb.DMatrix([sorted_inputs], feature_names=feature_names)
            prediction = model.predict(input_matrix, output_margin=False)

            # Prepare row for CSV
            row = {}
            if is_classification:
                for i, pred in enumerate(prediction[0]):
                    row[f'prediction_class_{i}'] = pred
            else:
                row['prediction'] = prediction[0]
                
            # Add features to row
            for name, value in inputs.items():
                row[name] = value
                
            writer.writerow(row)
    
    return worker_file

# Provided an XGBoost model and information about its columns, search
# all possible combinations of inputs and save the outputs to CSVs
def dataspace_search(model, dataspace, num_threads=max(1, cpu_count() - 1)):
    # Step 1: Warn user about the total combinations 
    print(f"Warning: The total number of combinations to search is {dataspace.size}.")
    print(f"Using {num_threads} threads for parallel processing. ({dataspace.size / num_threads} combinations per thread.)")

    # Step 2: Validate model and dataspace compatibility
    validate_columns(model, dataspace)

    # Step 3: Parallelized Search with CSV output
    worker_indices = dataspace.get_worker_indices(num_threads)
    worker_files = []
    output_file = "Dataspace.csv"

    def process_worker_range(args):
        worker_id, (start_index, num_combinations) = args
        return worker_search(model, dataspace, start_index, num_combinations, worker_id)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        worker_args = enumerate([
            (start_index, num_combinations)
            for start_index, num_combinations in worker_indices
        ])
        worker_files = list(executor.map(process_worker_range, worker_args))

    # Step 4: Combine Results
    print(f"Combining results from {len(worker_files)} workers into {output_file}...")
    
    # Copy header from first worker file
    with open(worker_files[0], 'r', newline='') as first_file:
        header = next(csv.reader(first_file))
    
    # Write combined results
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        # Process each worker file
        for worker_file in worker_files:
            with open(worker_file, 'r', newline='') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)
            
            # Clean up worker file
            os.remove(worker_file)
            
    print(f"Search complete. Results saved to {output_file}")

# Testing the Implementation
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
    ds.append(LinearDatarange('TimeOfDay', 0, 43200, 2))
    ds.append(LinearDatarange('GameOfDay', 1, 10, 2))
    ds.append(LinearDatarange('GameOfWeek', 1, 50, 2))
    ds.append(LinearDatarange('EloDifference', -10, 10, 3))
    ds.append(LinearDatarange('TimeSinceLast', 0, 158400, 2))

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

def main():
    test1()
    #test2()
    #test3()

if __name__ == "__main__":
    main()