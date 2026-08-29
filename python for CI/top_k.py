import heapq

# holds min heap of size three and checks if anything is bigger than the root of minheap
# O(n*lok(3))
def top_three_largest(nums: list[int]) -> list[int]:

    if len(nums) <= 3:
        return sorted(nums, reverse=True)

    minheap = nums[:3]
    heapq.heapify(minheap)


    for val in nums[3:]:
        if val > minheap[0]:
            heapq.heapreplace(minheap, val)

    return sorted(minheap, reverse=True)

