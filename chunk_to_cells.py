import os
import argparse
import numpy as np
from scipy.sparse import csr_matrix, save_npz

def extract_cells(input_dir, start_idx, end_idx, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the input directory
    files = sorted(f for f in os.listdir(input_dir) if f.startswith("cells_merged_") and f.endswith(".npz"))
    
    # Iterate over files, filter by start and end indices
    for file_name in files:
        # Parse start and end index from file name
        try:
            parts = file_name.replace("cells_merged_", "").replace(".npz", "").split("_to_")
            file_start = int(parts[0])
            file_end = int(parts[1])
        except (IndexError, ValueError):
            print(f"Skipping file {file_name}, unable to parse start and end indices.")
            continue
        
        # Check if this file contains cells in the specified range
        if file_start >= end_idx or file_end <= start_idx:
            continue  # Skip files outside the specified range
        
        # Load the sparse matrix from the file
        file_path = os.path.join(input_dir, file_name)
        data = np.load(file_path)
        sparse_matrix = csr_matrix((data['data'], data['indices'], data['indptr']), shape=(file_end - file_start, 27874))
        
        # Extract and save each row as a separate sparse matrix if it falls within the range
        for i in range(sparse_matrix.shape[0]):
            cell_idx = file_start + i
            if cell_idx >= start_idx and cell_idx < end_idx:
                cell_row = sparse_matrix.getrow(i)
                output_path = os.path.join(output_dir, f"cell{cell_idx}.npz")
                save_npz(output_path, cell_row)
                print(f"Saved cell {cell_idx} to {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract and save individual cells as sparse matrices from chunk files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the chunk files")
    parser.add_argument("--start_idx", type=int, required=True, help="Start index of cells to extract")
    parser.add_argument("--end_idx", type=int, required=True, help="End index of cells to extract (exclusive)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save individual cell files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run extraction
    extract_cells(args.input_dir, args.start_idx, args.end_idx, args.output_dir)