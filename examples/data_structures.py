# Data structures example
def process_list():
    """Example of list operations"""
    numbers = [1, 2, 3, 4, 5]
    doubled = [x * 2 for x in numbers]
    
    result = []
    for num in doubled:
        if num > 5:
            result.append(num)
    
    return result

def process_dict():
    """Example of dictionary operations"""
    data = {
        "name": "John",
        "age": 30,
        "city": "New York"
    }
    
    # Add new entry
    data["country"] = "USA"
    
    # Process values
    for key, value in data.items():
        print(f"{key}: {value}")
    
    return data

def string_operations():
    """Example of string operations"""
    text = "Hello, World!"
    
    # Basic operations
    upper_text = text.upper()
    lower_text = text.lower()
    split_text = text.split(", ")
    
    # String formatting
    name = "Python"
    version = 3.9
    message = f"Welcome to {name} {version}!"
    
    return {
        "original": text,
        "upper": upper_text,
        "lower": lower_text,
        "split": split_text,
        "formatted": message
    }

def control_flow_example():
    """Example of control flow statements"""
    results = []
    
    # For loop with range
    for i in range(10):
        if i % 2 == 0:
            results.append(f"Even: {i}")
        else:
            results.append(f"Odd: {i}")
    
    # While loop
    count = 0
    while count < 5:
        results.append(f"Count: {count}")
        count += 1
    
    return results

def error_handling_example():
    """Example of error handling"""
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        result = None
    except Exception as e:
        print(f"Unexpected error: {e}")
        result = None
    finally:
        print("Cleanup code here")
    
    return result

def main():
    """Main function"""
    print("List processing result:", process_list())
    print("\nDictionary processing:")
    process_dict()
    print("\nString operations:", string_operations())
    print("\nControl flow example:", control_flow_example())
    print("\nError handling example:", error_handling_example())

if __name__ == "__main__":
    main()
