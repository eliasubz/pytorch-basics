s = "abcdcba"

def trapping_rain_water(height: list[int]) -> int:
    if not height: 
        return 0

    l, r = 0, len(height)-1
    total = 0
    while (l < len(height)):
        cur = min(height[l], height[r])


def is_palindrom(s):
    l, r = 0, len(s)-1
    # palindrom
    while(l < r):
        if(s[l] != s[r]): return False
        l += 1
        r -= 1

    return True

def four_sum(nums: list[int], target: int) -> list[list[int]]:
    nums.sort()
    results = []
    n = len(nums)

    for i in range(n-3):
        if i > 0 and nums[i] == nums[i-1]:
            continue # skip outer duplicate

        # early termination
        if sum(nums[i : i + 4]) > target:
            break
        if nums[i] + sum(nums[-3:]) < target:
            continue

        for j in range(i + 1, n -2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue # skipping second outer duplicate

            l, r = j+1, n-1
            while l < r:
                cur_sum = nums[i] + nums[j] + nums[l] + nums[r]

                if cur_sum == target:
                    results.append([nums[i], nums[j], nums[l], nums[r]])

                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif cur_sum < target:
                    l += 1
                else:
                    r -= 1

        return results

        
nums = [ i for i in range(6)]
print(nums[0:4])
print(nums[::-1])

print(is_palindrom(s))
print("Hello world!")