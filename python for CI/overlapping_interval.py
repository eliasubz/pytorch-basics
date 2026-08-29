def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    # sort by end time
    sorted(intervals, key= lambda x: x[1])
    non_overlapping_count = 0
    prev_end = float("-inf")
    for start, end in intervals:
        if start >= prev_end:
            non_overlapping_count += 1
            prev_end = end

    return non_overlapping_count


import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0
    sorted(intervals, key=lambda x: x[0])

    minheap = []
    heapq.heappush(minheap, intervals[0][1])
    for start, end in intervals:
        if minheap[0] <= start:
            minheap.pop()
        heapq.heappush(minheap, end)

    return len(minheap)

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    # sort intervals by start
    sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        _, prev_end = merged[-1]

        if start <= prev_end:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return merged

