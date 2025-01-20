import logging
import time
from get_pgn import get_pgn
from pgn_to_csv import pgn_to_csv
from combine_csv import combine_csv
from feature_engineering import feature_engineering

def pipeline(usernames):
    """
    Runs the data pipeline for the given list of usernames.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pipeline.log"),
            logging.StreamHandler()
        ]
    )

    start_time = time.time()
    logging.info("Starting the pipeline process.")

    # Step 1: Run get_pgn.py for each username
    for username in usernames:
        logging.info(f"Starting get_pgn.py for username: {username}")
        user_start_time = time.time()
        get_pgn(username)
        logging.info(f"Completed get_pgn.py for username: {username} in {time.time() - user_start_time:.2f} seconds.")

    # Step 2: Run pgn_to_csv.py for each username
    for username in usernames:
        logging.info(f"Starting pgn_to_csv.py for username: {username}")
        user_start_time = time.time()
        pgn_to_csv(username)
        logging.info(f"Completed pgn_to_csv.py for username: {username} in {time.time() - user_start_time:.2f} seconds.")

    # Step 3: Run combine_csv.py with the list of usernames
    logging.info("Starting combine_csv.py with all usernames.")
    combine_start_time = time.time()
    combine_csv(usernames)
    logging.info(f"Completed combine_csv.py in {time.time() - combine_start_time:.2f} seconds.")

    # Step 4: Run feature_engineering.py
    logging.info("Starting feature_engineering.py.")
    feature_engineering_start_time = time.time()
    feature_engineering()
    logging.info(f"Completed feature_engineering.py in {time.time() - feature_engineering_start_time:.2f} seconds.")

    total_time = time.time() - start_time
    logging.info(f"Pipeline process completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    # Define the list of usernames
    usernames = ["Cosmos_IV", "CosmosSolitarus"]

    # Run the pipeline
    pipeline(usernames)
