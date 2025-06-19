import numpy as np

def process_arrays():
    """Process arrays using NumPy"""
    # Create arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.zeros(10)
    arr3 = np.ones(5)
    
    # Perform operations
    result = np.sum(arr1)
    mean_val = np.mean(arr1)
    
    print(f"Sum: {result}")
    print(f"Mean: {mean_val}")
    
    return result

if __name__ == "__main__":
    process_arrays()
