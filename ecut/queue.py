import heapq


class Entry:
    def __init__(self, task, cost):
        self.task = task
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __eq__(self, other):
        return self.cost == other.cost


class PriorityQueue:
    def __init__(self):
        self._queue: list[Entry] = []
        self._entry_finder: dict[int, Entry] = {}

    def empty(self):
        return len(self._queue) == 0

    def add_task(self, task, cost):
        """Add a new task or update the priority of an existing task"""
        if task in self._entry_finder:
            self._remove_task(task)
        entry = Entry(task, cost)
        self._entry_finder[task] = entry
        heapq.heappush(self._queue, entry)

    def _remove_task(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self._entry_finder.pop(task)
        entry.task = None

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self._queue:
            entry = heapq.heappop(self._queue)
            if entry.task is not None:
                del self._entry_finder[entry.task]
                return entry.task
        return None