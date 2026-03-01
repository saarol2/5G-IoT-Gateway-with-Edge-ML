import time
from collections import deque
from typing import Dict, List, Any

class GatewayMetrics:
    """Tracks gateway performance metrics for evaluation"""
    
    def __init__(self, gateway_id: str, report_interval: int = 10, warmup_duration: int = 60):
        self.gateway_id = gateway_id
        self.report_interval = report_interval
        self.warmup_duration = warmup_duration
        
        # Counters
        self.messages_received = 0
        self.messages_sent = 0
        self.messages_failed = 0
        self.predictions_made = 0
        self.anomalies_detected = 0
        
        # Latency tracking (rolling window of last 100)
        self.latencies = deque(maxlen=100)
        
        # Buffer usage tracking
        self.buffer_samples = deque(maxlen=100)
        
        # Timing
        self.start_time = time.time()
        self.last_report = time.time()
        self.warmup_complete = False
        
        # Batch tracking
        self.batch_count = 0
        self.total_batch_processing_time = 0
        
    def is_warming_up(self) -> bool:
        """Check if still in warmup period"""
        elapsed = time.time() - self.start_time
        if not self.warmup_complete and elapsed >= self.warmup_duration:
            self.warmup_complete = True
            print(f"\n[{self.gateway_id}] ✓ Warmup complete ({self.warmup_duration}s) - Starting metrics collection\n")
        return not self.warmup_complete
        
    def record_batch_received(self, count: int):
        """Record messages received in buffer"""
        if self.is_warming_up():
            return
        self.messages_received += count
        
    def record_batch_sent(self, count: int, processing_time: float):
        """Record successful batch send"""
        if self.is_warming_up():
            return
        self.messages_sent += count
        self.batch_count += 1
        self.total_batch_processing_time += processing_time
        
    def record_batch_failed(self, count: int):
        """Record failed batch send"""
        if self.is_warming_up():
            return
        self.messages_failed += count
        
    def record_prediction(self, is_anomaly: bool):
        """Record ML prediction"""
        if self.is_warming_up():
            return
        self.predictions_made += 1
        if is_anomaly:
            self.anomalies_detected += 1
            
    def record_latency(self, device_timestamp: float, cloud_timestamp: float):
        """Record end-to-end latency in milliseconds"""
        if self.is_warming_up():
            return
        latency_ms = (cloud_timestamp - device_timestamp) * 1000
        self.latencies.append(latency_ms)
        
    def record_buffer_usage(self, usage: float):
        """Record buffer usage percentage (0.0 - 1.0)"""
        if self.is_warming_up():
            return
        self.buffer_samples.append(usage)
        
    def should_report(self) -> bool:
        """Check if it's time to print report"""
        if self.is_warming_up():
            return False
        return (time.time() - self.last_report) >= self.report_interval
        
    def get_stats(self) -> Dict[str, Any]:
        """Calculate current statistics"""
        # Calculate elapsed time from end of warmup period
        if self.warmup_complete:
            elapsed = time.time() - (self.start_time + self.warmup_duration)
        else:
            elapsed = 0
        
        # Throughput
        msgs_per_sec = self.messages_sent / elapsed if elapsed > 0 else 0
        
        # Latency
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        
        # Buffer
        avg_buffer = sum(self.buffer_samples) / len(self.buffer_samples) if self.buffer_samples else 0
        max_buffer = max(self.buffer_samples) if self.buffer_samples else 0
        
        # Success rate
        total_attempted = self.messages_sent + self.messages_failed
        success_rate = (self.messages_sent / total_attempted * 100) if total_attempted > 0 else 0
        
        # Avg batch processing time
        avg_batch_time = (self.total_batch_processing_time / self.batch_count * 1000) if self.batch_count > 0 else 0
        
        return {
            "elapsed_sec": elapsed,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "throughput_msg_sec": msgs_per_sec,
            "success_rate_pct": success_rate,
            "predictions_made": self.predictions_made,
            "anomalies_detected": self.anomalies_detected,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_buffer_usage_pct": avg_buffer * 100,
            "max_buffer_usage_pct": max_buffer * 100,
            "avg_batch_processing_ms": avg_batch_time,
            "batches_sent": self.batch_count,
        }
        
    def print_report(self):
        """Print formatted metrics report"""
        stats = self.get_stats()
        self.last_report = time.time()
        
        print(f"\n{'='*70}")
        print(f"[METRICS] Gateway: {self.gateway_id} | Runtime: {stats['elapsed_sec']:.1f}s")
        print(f"{'='*70}")
        print(f"📊 THROUGHPUT:")
        print(f"  • Messages sent:     {stats['messages_sent']:,} ({stats['throughput_msg_sec']:.1f} msg/s)")
        print(f"  • Messages failed:   {stats['messages_failed']:,}")
        print(f"  • Success rate:      {stats['success_rate_pct']:.1f}%")
        print(f"  • Batches sent:      {stats['batches_sent']}")
        print(f"")
        print(f"⏱️  LATENCY:")
        print(f"  • Average:           {stats['avg_latency_ms']:.1f} ms")
        print(f"  • Min / Max:         {stats['min_latency_ms']:.1f} / {stats['max_latency_ms']:.1f} ms")
        print(f"  • Batch processing:  {stats['avg_batch_processing_ms']:.1f} ms")
        print(f"")
        print(f"📦 BUFFER:")
        print(f"  • Avg usage:         {stats['avg_buffer_usage_pct']:.1f}%")
        print(f"  • Max usage:         {stats['max_buffer_usage_pct']:.1f}%")
        print(f"")
        print(f"🤖 MACHINE LEARNING:")
        print(f"  • Predictions made:  {stats['predictions_made']:,}")
        print(f"  • Anomalies found:   {stats['anomalies_detected']:,} ({stats['anomalies_detected']/stats['predictions_made']*100 if stats['predictions_made'] > 0 else 0:.1f}%)")
        print(f"{'='*70}\n")
        
    def print_csv_header(self):
        """Print CSV header for logging"""
        print("timestamp,gateway_id,elapsed_sec,msgs_sent,msgs_failed,throughput_msg_sec,"
              "success_rate_pct,predictions,anomalies,avg_latency_ms,max_latency_ms,"
              "avg_buffer_pct,max_buffer_pct,batches_sent")
              
    def print_csv_row(self):
        """Print current stats as CSV row"""
        stats = self.get_stats()
        print(f"{time.time()},{self.gateway_id},{stats['elapsed_sec']:.1f},"
              f"{stats['messages_sent']},{stats['messages_failed']},"
              f"{stats['throughput_msg_sec']:.2f},{stats['success_rate_pct']:.2f},"
              f"{stats['predictions_made']},{stats['anomalies_detected']},"
              f"{stats['avg_latency_ms']:.2f},{stats['max_latency_ms']:.2f},"
              f"{stats['avg_buffer_usage_pct']:.2f},{stats['max_buffer_usage_pct']:.2f},"
              f"{stats['batches_sent']}")
