# Class example
class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        """Initialize the calculator"""
        self.result = 0
        self.history = []
    
    def add(self, a, b):
        """Add two numbers"""
        self.result = a + b
        self.history.append(f"{a} + {b} = {self.result}")
        return self.result
    
    def subtract(self, a, b):
        """Subtract two numbers"""
        self.result = a - b
        self.history.append(f"{a} - {b} = {self.result}")
        return self.result
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        self.result = a * b
        self.history.append(f"{a} * {b} = {self.result}")
        return self.result
    
    def divide(self, a, b):
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        self.result = a / b
        self.history.append(f"{a} / {b} = {self.result}")
        return self.result
    
    def get_result(self):
        """Get the last result"""
        return self.result
    
    def get_history(self):
        """Get calculation history"""
        return self.history
    
    def clear(self):
        """Clear result and history"""
        self.result = 0
        self.history = []

class ScientificCalculator(Calculator):
    """A scientific calculator that extends the basic calculator"""
    
    def __init__(self):
        super().__init__()
        self.memory = 0
    
    def power(self, base, exponent):
        """Calculate base to the power of exponent"""
        self.result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {self.result}")
        return self.result
    
    def square_root(self, n):
        """Calculate square root"""
        import math
        self.result = math.sqrt(n)
        self.history.append(f"sqrt({n}) = {self.result}")
        return self.result
    
    def store_memory(self):
        """Store current result in memory"""
        self.memory = self.result
    
    def recall_memory(self):
        """Recall value from memory"""
        return self.memory

def main():
    """Main function to test the calculator"""
    calc = Calculator()
    print("Basic Calculator:")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    
    sci_calc = ScientificCalculator()
    print("\nScientific Calculator:")
    print(f"2 ^ 8 = {sci_calc.power(2, 8)}")
    print(f"sqrt(16) = {sci_calc.square_root(16)}")
    
    print("\nHistory:")
    for entry in calc.get_history():
        print(f"  {entry}")

if __name__ == "__main__":
    main()
