import math 
from itertools import accumulate

arr = [3, 1, 4, 1, 5, 9]
print(arr)
# with itertools::accumulate
print(list(accumulate(arr)))


# with loop with append
prfs = list([0])
print(prfs)
for i, v in enumerate(arr):
    prfs.append(prfs[i] + v)
print(prfs)

# with loop but already an array and leading zero
prfs = [0] * (len(arr) + 1)
for i, v in enumerate(arr):
    prfs[i+1] = prfs[i] + v

print(prfs)
# with leading zero at i: v[i] is the sum until i without v[i]

# Use cases
from itertools import accumulate

# sum between L=1, R=4 
L, R = 1, 4
prfs = list(accumulate(arr))
# Inclusive
inc = prfs[R] - prfs[L-1] # at i: v is sum until i including i
print(inc)
# Exclusive
print(prfs[R-1] - prfs[L])

