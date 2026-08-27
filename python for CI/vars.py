print("Beginning of script")
class Solution:
    def checkPossibility(self):
        n, m, z = 1, 1.2, "abs"


n= 1
while n < 5:
    print(n)
    n += 1

for i in range(5):
    print(i)

for i in range(2, 6):
    print(i)

for i in range(5, 1, -1):
    print(i)

print(5 / 2)

print (5 // 2)

print (-3 // 2) # -> -2 !? 

print (int(-3 / 2)) # -> -1 towards 0

print(10 % 3) # -> 1

print(-10 % 3) # -> 2? 
import math 
print(math.fmod(-10, 3)) # -> 1

print(math.floor(3 / 2))
float("inf")
arr = [2, 3, 4]
print(arr)

arr.append(7)
arr.append(5)

print(arr)

arr.pop()
print(arr)

arr.insert(1, 3)
print(arr)

nums = [21] * 4
for i, n in enumerate(nums):
    print(i, n)

nums1 = [1, 3, 5]
nums2 = [2, 4, 6]

for a, b in zip(nums1, nums2):
    print(a, b)

nums.reverse()
nums.sort()
nums.sort(reverse=True)
print(nums)


arr = ["bob", "alice", "jane", "doe"]
arr.sort()
print(arr)

arr.sort(key=lambda x:len(x))

# list comprehensions
arr = [i*i for i in range(5)]
print(arr)

# 2D
arr = [[3, 2] * 2 for _ in range(4)]
print(arr)

# strings
s = "abc"
print(s[0:2])
s += "def"

print(int("123") + int("456"))
print(str(123) + str(456))

print(ord("a"))

strings = ["ab", "cd", "df"]
print("".join(strings))

# queues
from collections import deque
queue = deque()
queue.append(1)
queue.appendleft(3)

# HashSet
mySet = set()

mySet.add(1)
mySet.add(2)
print(mySet)
mySet.add(2)
print(len(mySet))
print(3 in mySet)
mySet.pop()
print(mySet)

# list to set
ass = set([i+1 for i in range(4)])
print(ass)
# set comprehension
ass = {i for i in range(4)}
print(ass)

# hashmap aka dict
myMap = {}
myMap["alice"] = 88

# hashMap comprehension¿
keys = [i for i in range(3)]
vals = ["amobi", "knapsack", "dieter"]
key_vals = [(k, v) for k, v in zip(keys, vals)]
mappi = {k :v for k, v in key_vals}
print(mappi)

myMap = dict()
myMap["Bob"] = 2
print(myMap)

# looping through a map
for k in mappi:
    print(k, mappi[k])

for v in mappi.values():
    print(v)

for k, v in mappi.items():
    print(k, v)

# tuples
tup = [i for i in range(10)]
print(tup[0::2])

myMap = {(1,2): 3}
print(myMap[(1,2)])

mySet = set()
mySet.add((1,2))
mySet.add((2,1))
print((1,2) in mySet)

# heaps
import heapq # are arrays under the hood (ofc)

min_heap = []
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 2)
heapq.heappush(min_heap, 4)
heapq.heappush(min_heap, 10)
heapq.heappush(min_heap, -1)
print(min_heap)
# min is always at zero
print(min_heap[0])
while len(min_heap):
    print(heapq.heappop(min_heap))

heapq.heapify(arr)

def outer(a, b):
    c = "c"

    def inner():
        return a + b + c
    return inner()

print(outer("a", "b"))

# nested functions nonlocal keyword
def double(arr, val):
    def helper():
        # modify array
        for i, n in enumerate(arr):
            arr[i] *= 2

        nonlocal val
        val *= 2
    helper()
    print(arr, val)

nums = [1, 2]
val = 3
double(nums, val)

class MyClass:
    def __init__(self, nums):
        # create member vars
        self.nums = nums
        self.size = len(nums)

    def getLength(self):
        return self.size

    def getDoubleLength(self):
        return 2 * self.getLength()



numbars = [i for i in range(5)]
a = MyClass(numbars)
print(a)
print(a.getLength())