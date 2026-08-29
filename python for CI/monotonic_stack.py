def next_greaters(nums: list[int]) -> int:
    n = len(nums)
    stack = []
    output = [-1] * n
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            output[stack.pop()] = nums[i]

        stack.append(i)
    return output
nums = [i for i in range(4)]
print(next_greaters(nums))
# wtf this is not intuitve at all
