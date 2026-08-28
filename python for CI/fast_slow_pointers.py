class ListNode:
    def __init__(self, val=0.0, next=None): 
        self.val = val
        self.next = next

def detect_cycle(head: ListNode | None) -> ListNode | None:

    # detect if a cycle exists
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            break
    else:
        return None

    # return entry
    entry = head
    while entry != slow:
        entry = entry.next
        slow = slow.next
    return entry

def find_middle(head: ListNode | None) -> ListNode | None:

    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    return slow

def is_happy(n: int) -> bool:
    def squaresum(val: int) -> int:
        total = 0
        while val > 0:
            val, digit = divmod(val, 10)
            total += digit * digit
        return total

    slow = n
    fast = squaresum(n)
    while fast != 1 and slow != fast:
        slow = squaresum(slow)
        fast = squaresum(squaresum(fast))
    return (fast == 1)