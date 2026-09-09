---
author: ["Abdullah Al Mamun"]
title: "Python DSA Cheatsheet For Coding Interview"
date: 2026-08-25
draft: false
comments: true
ShowToc: true
TocOpen: false
slug: "python-dsa-leetcode-cheatsheet"
description: "A compact Python syntax reference for LeetCode-style coding interviews: core syntax, hashmaps, sets, sorting, and the DSA patterns (two pointers, sliding window, stack, BFS, heap, binary search, graphs) worth committing to muscle memory."
summary: "Python syntax and DSA patterns for coding interviews, condensed into a single reference: core syntax on page 1, the coding patterns worth memorizing on page 2."
keywords:
  - "Python"
  - "LeetCode"
  - "DSA"
  - "coding interview"
  - "algorithms"
  - "data structures"
tags:
  - "python"
  - "leetcode"
  - "dsa"
  - "interview prep"
  - "algorithms"
categories:
  - "Engineering Notes"
---

> **Goal:** Python syntax for LeetCode-style interviews.  
> **Focus:** essential syntax and patterns, not a comprehensive Python reference.

---

## PAGE 1 — Core Python Syntax

### 1. Variables & Basic Operations

```python
x = 5
a, b = 1, 2
a, b = b, a                  # swap

x += 1
x -= 1
x *= 2
x //= 2                     # integer division

abs(x)
min(a, b)
max(a, b)
sum(nums)
```

---

### 2. Strings

```python
s = "Hello"

len(s)
s[i]                        # character
s[i:j]                      # substring
s[::-1]                     # reverse

s.lower()
s.upper()
s.strip()

s.split()                   # "a b c" -> ["a", "b", "c"]
" ".join(words)             # ["a", "b"] -> "a b"
"".join(sorted(s))          # sorted characters

s.isalpha()
s.isdigit()
s.isalnum()
```

#### Character ↔ ASCII

```python
ord('a')                    # 97
chr(97)                     # 'a'
```

#### Common string loops

```python
for c in s:
    ...

for i, c in enumerate(s):
    ...
```

---

### 3. Lists / Arrays

```python
nums = []
nums = [1, 2, 3]

len(nums)

nums.append(x)
nums.pop()                  # remove last
nums.pop(i)                 # remove index i

nums[i]
nums[i:j]

nums.reverse()              # in-place
nums.sort()                 # in-place

sorted(nums)                # returns new list
```

#### Useful shortcuts

```python
nums = [0] * n              # [0, 0, 0, ...]

nums = list(range(n))       # [0, 1, 2, ..., n-1]

nums = list(range(1, n + 1))

x = nums[-1]                # last
x = nums[-2]                # second last
```

#### Looping

```python
for x in nums:
    ...

for i in range(len(nums)):
    ...

for i, x in enumerate(nums):
    ...

for a, b in zip(nums1, nums2):
    ...
```

#### List comprehension

```python
squares = [x * x for x in nums]

even = [x for x in nums if x % 2 == 0]

zeros = [0 for _ in range(n)]
```

---

### 4. HashMap / Dictionary ⭐

#### Initialize

```python
mp = {}
```

#### Insert / update

```python
mp[key] = value

mp[key] += 1
```

If key might not exist:

```python
mp[key] = mp.get(key, 0) + 1
```

#### Check

```python
if key in mp:
    ...

if key not in mp:
    ...
```

#### Get

```python
value = mp[key]

value = mp.get(key)         # None if missing
value = mp.get(key, 0)      # 0 if missing
```

#### Loop

```python
for key in mp:
    ...

for key, value in mp.items():
    ...
```

#### Frequency Map

```python
count = {}

for x in nums:
    count[x] = count.get(x, 0) + 1
```

#### Dictionary comprehension

```python
mp = {x: 0 for x in nums}
```

---

### 5. Set ⭐

```python
seen = set()

seen.add(x)
seen.remove(x)              # must exist
seen.discard(x)             # safe if missing

if x in seen:
    ...

len(seen)
```

#### Convert

```python
set(nums)
list(set(nums))
```

#### Classic "seen" pattern

```python
seen = set()

for x in nums:
    if x in seen:
        return True
    seen.add(x)
```

---

### 6. Sorting ⭐

```python
nums.sort()                 # modify nums
nums.sort(reverse=True)

sorted_nums = sorted(nums)

sorted(nums, reverse=True)
```

#### Sort with key

```python
items.sort(key=lambda x: x[0])

items.sort(key=lambda x: x[1], reverse=True)

sorted(items, key=lambda x: x[0])
```

#### Sort by multiple fields

```python
items.sort(key=lambda x: (x[0], -x[1]))
```

#### Zip + sort + unpack

```python
time_sorted = sorted(
    zip(position, speed),
    key=lambda x: x[0],
    reverse=True
)

position, speed = zip(*time_sorted)
```

---

### 7. Tuples

```python
t = (1, 2)

a, b = t

x, y = y, x
```

#### Useful for coordinates

```python
r, c = (2, 3)

queue.append((r, c))
```

#### Unpacking

```python
for r, c in points:
    ...
```

---

### 8. Conditionals / One-Liners

```python
if x > 0:
    ...

if x > 0:
    ...
else:
    ...
```

#### Ternary

```python
x = a if condition else b
```

Example:

```python
mx = a if a > b else b
```

#### Multiple conditions

```python
if x > 0 and y > 0:
    ...

if x == 0 or y == 0:
    ...

if not seen:
    ...
```

---

## PAGE 2 — DSA Coding Patterns

### 9. Two Pointers ⭐

```python
left = 0
right = len(nums) - 1

while left < right:
    if nums[left] + nums[right] == target:
        return [left, right]
    elif nums[left] + nums[right] < target:
        left += 1
    else:
        right -= 1
```

#### String version

```python
left = 0
right = len(s) - 1

while left < right:
    ...
    left += 1
    right -= 1
```

---

### 10. Sliding Window ⭐

#### Basic template

```python
left = 0

for right in range(len(nums)):
    # add nums[right]

    while condition:
        # remove nums[left]
        left += 1

    # update answer
```

#### With HashMap

```python
window = {}
left = 0

for right, c in enumerate(s):
    window[c] = window.get(c, 0) + 1

    while ...:
        window[s[left]] -= 1
        left += 1
```

---

### 11. Stack ⭐

Python `list` = stack.

```python
stack = []

stack.append(x)             # push
x = stack.pop()             # pop
x = stack[-1]               # peek

if stack:
    ...
```

#### Common monotonic-stack pattern

```python
for x in nums:
    while stack and stack[-1] < x:
        stack.pop()

    stack.append(x)
```

---

### 12. Queue / BFS ⭐

**Don't use `pop(0)` — it is O(n).**

```python
from collections import deque

q = deque()

q.append(x)
x = q.popleft()
```

#### BFS

```python
q = deque([start])

while q:
    node = q.popleft()

    for nei in graph[node]:
        q.append(nei)
```

#### Grid BFS directions

```python
dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

for dr, dc in dirs:
    nr = r + dr
    nc = c + dc
```

---

### 13. Heap / Priority Queue ⭐

Python provides a **min heap**.

```python
import heapq

heap = []

heapq.heappush(heap, x)
x = heapq.heappop(heap)

x = heap[0]                 # smallest
```

#### Build heap

```python
heapq.heapify(nums)
```

#### Max heap

Negate values:

```python
heapq.heappush(heap, -x)

x = -heapq.heappop(heap)
```

#### Heap of tuples

```python
heapq.heappush(heap, (priority, value))
```

Python compares tuples from left to right, so `priority` is used first.

---

### 14. Binary Search ⭐

#### Basic template

```python
left = 0
right = len(nums) - 1

while left <= right:
    mid = (left + right) // 2

    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

return -1
```

#### Python shortcut

```python
import bisect

i = bisect.bisect_left(nums, target)
```

`bisect_left` = first position where `target` can be inserted.

---

### 15. Graph

#### Adjacency List

```python
graph = {}

graph[u] = []
graph[u].append(v)
```

Usually easier:

```python
from collections import defaultdict

graph = defaultdict(list)

graph[u].append(v)
graph[v].append(u)
```

#### Recursive DFS

```python
def dfs(node):
    if node in seen:
        return

    seen.add(node)

    for nei in graph[node]:
        dfs(nei)
```

#### Iterative DFS

```python
stack = [start]
seen = set()

while stack:
    node = stack.pop()

    if node in seen:
        continue

    seen.add(node)

    for nei in graph[node]:
        stack.append(nei)
```

---

### 16. Memoization / DP with `@lru_cache` ⭐

Top-down DP is just recursion plus a cache. `@lru_cache(None)` gives you the cache for free, so you write the plain recurrence and let Python remember the answers.

```python
from functools import lru_cache

@lru_cache(None)            # None = unbounded cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```

Without the decorator this recomputes the same subproblems exponentially many times. With it, each `n` is computed once, so it runs in O(n).

#### Python 3.9+ shortcut

```python
from functools import cache

@cache                      # identical to @lru_cache(None)
def fib(n):
    ...
```

#### Inside a class (the interview pattern)

Decorating a method works, but `self` becomes part of the cache key and keeps
every instance alive. Nest the helper instead:

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        @lru_cache(None)
        def dp(i):
            if i <= 2:
                return i
            return dp(i - 1) + dp(i - 2)
        return dp(n)
```

#### Grid DP

```python
@lru_cache(None)
def dp(r, c):
    if r == 0 or c == 0:
        return 1
    return dp(r - 1, c) + dp(r, c - 1)
```

#### Gotchas

- **Arguments must be hashable.** Pass ints or tuples, never lists or dicts.
  Convert at the boundary: `dp(tuple(nums), 0)`.
- **A module-level cache survives between test cases.** Reset it with
  `dp.cache_clear()` if stale results could leak across runs.
- `dp.cache_info()` shows hits and misses, which is the fastest way to confirm
  the memo is actually being used.

---

### 17. Common `collections` ⭐

```python
from collections import Counter, defaultdict, deque
```

#### Counter

```python
count = Counter(s)

count['a']

count.most_common(1)
```

Instead of:

```python
count = {}

for c in s:
    count[c] = count.get(c, 0) + 1
```

#### defaultdict

```python
graph = defaultdict(list)

graph[x].append(y)
```

#### deque

```python
q = deque()

q.append(x)
q.popleft()
```

---

### 18. Useful Built-ins

```python
len(nums)
sum(nums)
min(nums)
max(nums)

any(condition for x in nums)
all(condition for x in nums)

enumerate(nums)
zip(a, b)

range(n)
range(start, end)
range(start, end, step)
```

#### Examples

```python
any(x > 10 for x in nums)

all(x >= 0 for x in nums)
```

---

## 19. The 10 Syntax Patterns to Memorize First ⭐⭐⭐

If you're just getting back into coding, **memorize these first**. Don't try to memorize the whole sheet.

```python
# 1. HashMap
mp = {}
mp[x] = mp.get(x, 0) + 1

# 2. Set
seen = set()
seen.add(x)
if x in seen:
    ...

# 3. Loop with index
for i, x in enumerate(nums):
    ...

# 4. Reverse
s[::-1]

# 5. Sort
nums.sort()
sorted(nums)

# 6. Sort with key
sorted(items, key=lambda x: x[1])

# 7. Stack
stack.append(x)
stack.pop()

# 8. Queue
from collections import deque
q = deque()
q.append(x)
q.popleft()

# 9. Heap
import heapq
heapq.heappush(heap, x)
heapq.heappop(heap)

# 10. Two pointers
left, right = 0, len(nums) - 1

while left < right:
    ...
    left += 1
    right -= 1
```

---

## 20. Three Loop Patterns You Should Instantly Recognize

#### Iterate values

```python
for x in nums:
    ...
```

#### Iterate indexes

```python
for i in range(len(nums)):
    ...
```

#### Iterate index + value

```python
for i, x in enumerate(nums):
    ...
```

---

## 21. Quick Mental Reference

When you're staring at a LeetCode problem and can't remember syntax:

| Need | Python |
|---|---|
| HashMap | `mp = {}` |
| HashMap frequency | `mp[x] = mp.get(x, 0) + 1` |
| Set | `seen = set()` |
| Add to set | `seen.add(x)` |
| Check set | `if x in seen:` |
| Stack | `stack = []` |
| Push | `stack.append(x)` |
| Pop | `stack.pop()` |
| Queue | `q = deque()` |
| Enqueue | `q.append(x)` |
| Dequeue | `q.popleft()` |
| Heap | `heapq.heappush(heap, x)` |
| Remove heap | `heapq.heappop(heap)` |
| Sort | `sorted(nums)` |
| In-place sort | `nums.sort()` |
| Reverse | `s[::-1]` |
| Length | `len(x)` |
| Index + value | `enumerate(nums)` |
| Pair arrays | `zip(a, b)` |
| Integer division | `a // b` |
| Remainder | `a % b` |
| First/last | `nums[0]`, `nums[-1]` |
| Subarray | `nums[l:r]` |
| Range | `range(n)` |
| Min/max | `min()`, `max()` |
| Absolute | `abs(x)` |
| Memoize recursion | `@lru_cache(None)` |

---

## Final Rule

**Don't expand this cheat sheet too quickly.**

You already know the DSA concepts. Your current bottleneck is **Python muscle memory**.

Practice like this:

```text
Problem
   ↓
Identify DSA pattern
   ↓
Open cheatsheet only for syntax
   ↓
Write code yourself
   ↓
Get stuck
   ↓
Add ONLY that missing syntax/pattern
```

The objective is to eventually stop looking at the sheet — not to build the world's most impressive Python cheat sheet.
