#import numpy as np
#from numpy import random


import sys


#arr = random.randint(100, size=(10))


# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

print(f'Before sorting {my_numbers}')


def partition(arr, l, h):
	pi=arr[h]
	i=(l-1)
	for j in range(l, h):
		if(arr[j]<pi):
			i=i+1
			s=arr[i]
			arr[i]=arr[j]
			arr[j]=s


	s=arr[i+1]
	arr[i+1]=arr[h]
	arr[h]=s
	return i+1

def quicksort(arr, l, h):
	if(l<h):
		pi=partition(arr, l, h)
		quicksort(arr, l,pi-1)
		quicksort(arr, pi+1, h)
	else:
		return arr


quicksort(my_numbers, 0, len(my_numbers)-1)
print(f'After sorting {my_numbers}')

