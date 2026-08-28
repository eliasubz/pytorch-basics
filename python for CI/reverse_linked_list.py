class ListNode:
    def __init__(self, val=0.0, next=None):
        self.val = val
        self.next = next

def reverse_LL(head: ListNode | None) -> ListNode | None:
    prev = None
    cur = head
    while cur.next != None:
        # store helper function
        loc_next = cur.next
        # set next of cur to prev (the reverse)
        cur.next = prev
        # set cur to logical next
        cur = loc_next
        # set prev to cur
        prev = cur

    return prev.next
