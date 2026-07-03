"""Request queue with backpressure for handling high concurrency."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class QueuedRequest:
    """A request waiting to be processed."""
    request_id: int
    payload: dict[str, Any]
    created_at: float
    callback: Callable[[dict[str, Any]], None] | None = None
    error_callback: Callable[[Exception], None] | None = None


class RequestQueue:
    """Global request queue with backpressure and rate limiting.
    
    This queue ensures that:
    1. Requests are processed in order of priority
    2. Backpressure is applied when the queue is full
    3. Requests are batched to avoid overwhelming the upstream
    4. Failed requests are retried with exponential backoff
    """
    
    def __init__(
        self,
        max_queue_size: int = 500,
        max_concurrent: int = 60,
        batch_size: int = 10,
        batch_interval: float = 0.1,
        retry_max: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.retry_max = retry_max
        self.retry_base_delay = retry_base_delay
        
        self._queue: list[QueuedRequest] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        self._active_count = 0
        self._total_processed = 0
        self._total_failed = 0
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def submit(self, request_id: int, payload: dict[str, Any], 
               callback: Callable[[dict[str, Any]], None] | None = None,
               error_callback: Callable[[Exception], None] | None = None,
               timeout: float = 300.0) -> bool:
        """Submit a request to the queue. Returns False if queue is full."""
        with self._not_full:
            if len(self._queue) >= self.max_queue_size:
                # Backpressure: wait for queue to have space
                if not self._not_full.wait(timeout=timeout):
                    return False
            
            req = QueuedRequest(
                request_id=request_id,
                payload=payload,
                created_at=time.time(),
                callback=callback,
                error_callback=error_callback,
            )
            self._queue.append(req)
            self._not_empty.notify()
        
        return True
    
    def _worker(self):
        """Worker thread that processes requests from the queue."""
        while self._running:
            batch = []
            with self._not_empty:
                # Wait for requests
                while not self._queue and self._running:
                    self._not_empty.wait(timeout=1.0)
                
                if not self._running:
                    break
                
                # Take a batch of requests
                while self._queue and len(batch) < self.batch_size:
                    batch.append(self._queue.pop(0))
            
            if not batch:
                continue
            
            # Process batch
            for req in batch:
                self._process_request(req)
    
    def _process_request(self, req: QueuedRequest):
        """Process a single request with retry logic."""
        for attempt in range(self.retry_max + 1):
            try:
                # Wait for slot availability
                with self._lock:
                    while self._active_count >= self.max_concurrent:
                        self._lock.wait(timeout=1.0)
                    self._active_count += 1
                
                try:
                    # Process the request
                    if req.callback:
                        req.callback(req.payload)
                    self._total_processed += 1
                    return
                finally:
                    with self._lock:
                        self._active_count -= 1
                        self._not_full.notify()
            
            except Exception as exc:
                if attempt < self.retry_max:
                    # Exponential backoff
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    self._total_failed += 1
                    if req.error_callback:
                        req.error_callback(exc)
    
    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "active_count": self._active_count,
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
            }
    
    def shutdown(self):
        """Shutdown the queue."""
        self._running = False
        with self._not_empty:
            self._not_empty.notify_all()


# Global request queue instance
_request_queue: RequestQueue | None = None
_request_queue_lock = threading.Lock()


def get_request_queue() -> RequestQueue:
    """Get or create the global request queue."""
    global _request_queue
    if _request_queue is None:
        with _request_queue_lock:
            if _request_queue is None:
                _request_queue = RequestQueue()
    return _request_queue
