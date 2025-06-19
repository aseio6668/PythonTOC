# Simple function example
def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def main():
    """Main function"""
    print("Fibonacci of 10:", fibonacci(10))
    print("Factorial of 5:", factorial(5))

if __name__ == "__main__":
    main()
