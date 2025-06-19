import math
from typing import List, Dict

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.data: List[float] = []
    
    def add_data(self, values: List[float]) -> None:
        self.data.extend(values)
    
    def calculate_statistics(self) -> Dict[str, float]:
        if not self.data:
            return {}
        
        n = len(self.data)
        mean = sum(self.data) / n
        variance = sum((x - mean) ** 2 for x in self.data) / n
        std_dev = math.sqrt(variance)
        
        return {
            "count": n,
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "min": min(self.data),
            "max": max(self.data)
        }

def main():
    processor = DataProcessor("Test Processor")
    processor.add_data([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = processor.calculate_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()