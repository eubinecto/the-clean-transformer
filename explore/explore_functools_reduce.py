import functools

# initializing list
lis = [1, 3, 2, 2, 4, 7, 3, 4]

# using reduce to compute sum of list
print(list(map(lambda a, b: a if a == 2 else b, lis, lis[1:])))
