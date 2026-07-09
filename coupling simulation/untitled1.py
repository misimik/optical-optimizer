# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:06:33 2024

@author: Michal
"""

def partition(numbers, num_sets):
    numbers.sort(reverse=True)
    sets = [[] for _ in range(num_sets)]
    sums = [0] * num_sets
    
    for num in numbers:
        min_sum_index = sums.index(min(sums))
        sets[min_sum_index].append(num)
        sums[min_sum_index] += num
    
    return sets

# Example usage:
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_sets = 3
result_sets = partition(numbers, num_sets)
for i, s in enumerate(result_sets):
    print(f"Set {i+1}: {s}")