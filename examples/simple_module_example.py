#!/usr/bin/env python3
"""
Test the dynamic module analysis system
"""

import math
import random
from typing import List, Dict

class SimpleCalculator:
    """A simple calculator class for testing dynamic analysis"""
    
    def __init__(self, precision: int = 2):
        self.precision = precision
        self.history: List[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return round(result, self.precision)
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return round(result, self.precision)
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent"""
        result = math.pow(base, exponent)
        self.history.append(f"{base} ^ {exponent} = {result}")
        return round(result, self.precision)
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()


def generate_random_numbers(count: int = 10) -> List[float]:
    """Generate a list of random numbers"""
    return [random.uniform(0, 100) for _ in range(count)]


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers"""
    if not numbers:
        return {}
    
    return {
        'mean': sum(numbers) / len(numbers),
        'min': min(numbers),
        'max': max(numbers),
        'sum': sum(numbers)
    }


def fibonacci(n: int) -> int:
    """Calculate fibonacci number (recursive implementation)"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Example usage
if __name__ == "__main__":
    calc = SimpleCalculator()
    
    # Perform some calculations
    result1 = calc.add(10, 5)
    result2 = calc.multiply(3, 4)
    result3 = calc.power(2, 8)
    
    # Generate and analyze random numbers
    numbers = generate_random_numbers(5)
    stats = calculate_statistics(numbers)
    
    # Calculate fibonacci
    fib_10 = fibonacci(10)
    
    print(f"Calculator results: {result1}, {result2}, {result3}")
    print(f"Random numbers: {numbers}")
    print(f"Statistics: {stats}")
    print(f"Fibonacci(10): {fib_10}")
    print(f"History: {calc.get_history()}")
