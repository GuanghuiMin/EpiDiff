import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

def create_dynamic_adjacency_matrix():
    """
    Reads daily state-to-state mobility data from CSV files and creates a 
    dynamic adjacency matrix.

    The script iterates through a date range, reads a corresponding mobility file for each day,
    and builds a 3D NumPy array of shape (T, N, N), where T is the number of days
    and N is the number of states.
    """
    
    # --- Configuration ---
    DATA_DIR = '/home/guanghui/DiffODE/data/dataset/COVID/state/mobility' # Directory containing the CSV files
    START_DATE = date(2020, 4, 15)
    END_DATE = date(2021, 4, 15)
    OUTPUT_FILENAME = '/home/guanghui/DiffODE/data/dataset/COVID/dynamic_adj.npy'

    # --- State and ID Mapping ---
    # Create a comprehensive list of all state FIPS codes you expect to see.
    # This ensures a consistent size for the matrix (N x N).
    # Using a predefined list is more robust than inferring from a single file.
    all_fips_codes = [
        '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', 
        '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', 
        '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', 
        '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', 
        '54', '55', '56', '72'
    ]
    
    # Create a mapping from FIPS code to matrix index
    fips_to_index = {fips: i for i, fips in enumerate(all_fips_codes)}
    num_states = len(all_fips_codes)

    # --- Main Processing Loop ---
    date_range = [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]
    all_matrices = []

    print(f"Processing data from {START_DATE} to {END_DATE}...")

    for current_date in date_range:
        # Construct filename based on date
        date_str = current_date.strftime('%Y_%m_%d')
        filename = f"daily_state2state_{date_str}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found for date {current_date}, skipping.")
            continue

        try:
            # Read the CSV file for the current day
            df = pd.read_csv(filepath)
            
            # Initialize a zero matrix for the current day
            adj_matrix = np.zeros((num_states, num_states), dtype=np.float32)

            # Populate the matrix with visitor flows
            for index, row in df.iterrows():
                origin_fips = str(row['geoid_o']).zfill(2)
                dest_fips = str(row['geoid_d']).zfill(2)
                
                # Use the 'visitor_flows' column for the edge weight
                flow = row['visitor_flows']

                # Check if FIPS codes are in our mapping
                if origin_fips in fips_to_index and dest_fips in fips_to_index:
                    origin_idx = fips_to_index[origin_fips]
                    dest_idx = fips_to_index[dest_fips]
                    adj_matrix[origin_idx, dest_idx] = flow/1e6  # Scale down by 1e6 for better numerical stability
                
            all_matrices.append(adj_matrix)
            print(f"Successfully processed {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # --- Save the final array ---
    if all_matrices:
        dynamic_adj_array = np.stack(all_matrices, axis=0)
        np.save(OUTPUT_FILENAME, dynamic_adj_array)
        print(f"\nSuccessfully created '{OUTPUT_FILENAME}' with shape: {dynamic_adj_array.shape}")
    else:
        print("\nNo data was processed. Output file not created.")


if __name__ == "__main__":
    create_dynamic_adjacency_matrix()