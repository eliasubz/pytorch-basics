stack = []

def next_greater(nums: list[int]) -> int:
    # find the next greater elem for each item
    n = len(nums)
    stack = []
    result = [-1] * n

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]

        stack.append(i)
    return result

# wtf this is not intuitve at all
