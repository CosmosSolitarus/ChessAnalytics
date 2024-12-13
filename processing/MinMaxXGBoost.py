import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import xgboost as xgb
import numpy as np

# Defines the bounds and precision of a parameter
# The parameter can be incremented by the step size from
# min to max to evaluate how an XGBoost's prediction changes
class Datarange:
    def __init__(self, name, min_value, max_value, steps):
        self.name = name
        self.min = min_value
        self.max = max_value
        self.steps = steps      # Steps is assumed to be an integer >= 2
        self.step_size = (max_value - min_value) / (steps - 1)
        self.type = 'numeric'

# Similar to a Datarange, but specifically for one-hot encoded
# categorical parameters
class OneHotDatarange:
    def __init__(self, names):
        self.names = names
        self.type = 'onehot'
        self.steps = len(names)

# A collection of Dataranges and OneHotDataranges. Intended to
# represent the entire search space for a set of parameters
# in an XGBoost model
class Dataspace:
    def __init__(self):
        self.ranges = []

    # Appends a datarange to the ranges list
    def append(self, datarange):
        self.ranges.append(datarange)

    # Calculates the total number of combinations across all ranges
    def size(self):
        total_size = 1

        for range_obj in self.ranges:
            total_size *= range_obj.steps
        
        return total_size

    # Convert a given index to a specific combination of values
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

    # Divides the total combinations evenly among a number of workers
    # The last worker is assigned remainder combinations, with a worst
    # case of num_workers - 1 extra combinations to check (trivial)
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

# Determines whether the model's columns match the dataspace's
# columns, and there are no extras for either
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

    print("Column validation passed.")

# Worker function to evaluate combinations
def workerSearch(model, dataspace, start_index, num_combinations):
    results = []
    feature_names = model.feature_names

    for idx in range(start_index, start_index + num_combinations):
        combination = dataspace.index_to_combination(idx)

        # Convert combination into a dictionary format for easier readability
        inputs = {}
        feature_idx = 0
        for range_obj in dataspace.ranges:
            if feature_idx % 100 == 0:
                print(feature_idx)
            
            if range_obj.type == 'numeric':
                inputs[range_obj.name] = combination[feature_idx]
                feature_idx += 1
            elif range_obj.type == 'onehot':
                for name in range_obj.names:
                    inputs[name] = combination[feature_idx]
                    feature_idx += 1

        # Sort the inputs dictionary by the feature names to match the model's expected order
        sorted_inputs = [inputs[name] for name in feature_names]

        # Create input matrix for prediction
        input_matrix = xgb.DMatrix([sorted_inputs], feature_names=feature_names)

        # Get prediction
        prediction = model.predict(input_matrix, output_margin=False)

        # If classification, gather probabilities for all classes
        if len(prediction.shape) > 1:
            probabilities = prediction[0].tolist()
            results.append({
                "inputs": inputs,
                "prediction": probabilities,
            })
        else:  # If regression
            results.append({
                "inputs": inputs,
                "prediction": prediction[0],
            })

    return results

# Provided an XGBoost model and information about its columns, search
# all possible combinations of inputs to determine the top_n best and
# worst case scenarios
def MinMaxSearch(model, dataspace, top_n=1, num_threads=max(1, cpu_count() - 1)):
    # Step 1: Warn user about the total combinations
    total_combinations = dataspace.size()
    print(f"Warning: The total number of combinations to search is {total_combinations}.")
    print(f"Using {num_threads} threads for parallel processing. ({total_combinations / num_threads} combinations per thread.)")

    # Step 2: Validate model and dataspace compatibility
    validate_columns(model, dataspace)

    # Step 3: Parallelized Search
    worker_indices = dataspace.get_worker_indices(num_threads)
    all_results = []

    def process_worker_range(worker_args):
        return workerSearch(*worker_args)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        worker_args = [
            (model, dataspace, start_index, num_combinations)
            for start_index, num_combinations in worker_indices
        ]
        results = list(executor.map(process_worker_range, worker_args))

        # Flatten results from all workers
        for worker_result in results:
            all_results.extend(worker_result)

    # Step 4: Combine Results and Format Output
    if len(all_results[0]["prediction"]) > 1:  # Classification
        class_results = {}
        num_classes = len(all_results[0]["prediction"])

        for class_idx in range(num_classes):
            class_name = f"Class_{class_idx}"
            sorted_results = sorted(
                all_results,
                key=lambda x: x["prediction"][class_idx]
            )
            class_results[class_name] = {
                "min": [
                    {
                        "likelihood": res["prediction"][class_idx],
                        "inputs": res["inputs"],
                        "all_class_likelihoods": {
                            f"Class_{i}": res["prediction"][i]
                            for i in range(num_classes)
                        },
                    }
                    for res in sorted_results[:top_n]
                ],
                "max": [
                    {
                        "likelihood": res["prediction"][class_idx],
                        "inputs": res["inputs"],
                        "all_class_likelihoods": {
                            f"Class_{i}": res["prediction"][i]
                            for i in range(num_classes)
                        },
                    }
                    for res in sorted_results[-top_n:]
                ],
            }
        output_data = class_results
    else:  # Regression
        sorted_results = sorted(all_results, key=lambda x: x["prediction"])
        output_data = {
            "min": [
                {
                    "value": res["prediction"],
                    "inputs": res["inputs"],
                }
                for res in sorted_results[:top_n]
            ],
            "max": [
                {
                    "value": res["prediction"],
                    "inputs": res["inputs"],
                }
                for res in sorted_results[-top_n:]
            ],
        }

    # Step 5: Output Results to JSON
    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {output_file}.")

# Testing the Implementation
def test1():
    # Load an XGBoost model from a saved JSON file
    model = xgb.Booster()
    model.load_model("json/model.json")

    # Create dataspace
    dataspace = Dataspace()

    # Add one-hot encoded columns for days of the week (Sunday through Friday)
    dataspace.append(OneHotDatarange(['IsSunday', 'IsMonday', 'IsTuesday', 'IsWednesday', 'IsThursday', 'IsFriday', 'IsSaturday']))

    # Add binary columns for result-related features
    dataspace.append(Datarange('LastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    dataspace.append(Datarange('LastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    dataspace.append(Datarange('LastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1
    dataspace.append(Datarange('2ndLastResultIsWin', 0, 1, 2))  # Binary: 0 or 1
    dataspace.append(Datarange('2ndLastResultIsDraw', 0, 1, 2))  # Binary: 0 or 1
    dataspace.append(Datarange('2ndLastResultIsLoss', 0, 1, 2))  # Binary: 0 or 1

    # Add continuous columns with defined ranges and steps
    dataspace.append(Datarange('TimeOfDay', 0, 43200, 2))
    dataspace.append(Datarange('GameOfDay', 1, 10, 2))
    dataspace.append(Datarange('GameOfWeek', 1, 50, 2))
    dataspace.append(Datarange('EloDifference', -10, 10, 3))
    dataspace.append(Datarange('TimeSinceLast', 0, 158400, 2))

    # Run MinMaxSearch
    MinMaxSearch(model, dataspace, top_n=1, num_threads=7)

def test2():
    # Load the XGBoost model (ensure this path is correct)
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

def main():
    test1()
    #test2()

if __name__ == "__main__":
    main()