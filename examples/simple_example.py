def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()