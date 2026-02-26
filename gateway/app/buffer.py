from collections import deque
import threading
from typing import Any, List

class ReadingBuffer:
    def __init__(self, maxlen: int):
        self._q = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.maxlen = maxlen

    def append(self, item: Any) -> None:
        with self._lock:
            self._q.append(item)

    def size(self) -> int:
        with self._lock:
            return len(self._q)

    def usage(self) -> float:
        with self._lock:
            return len(self._q) / self.maxlen

    def peek_batch(self, n: int) -> List[Any]:
        with self._lock:
            return list(self._q)[:n]

    def drop(self, n: int) -> None:
        with self._lock:
            for _ in range(min(n, len(self._q))):
                self._q.popleft()
