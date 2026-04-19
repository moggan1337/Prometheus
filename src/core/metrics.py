"""
Metrics Collector Module - Tracks and aggregates evolution metrics.

Collects various metrics during evolution:
- Fitness progression
- Diversity trends
- Mutation rates
- Convergence indicators
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict
from pathlib import Path


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    generation: int
    value: float
    metadata: dict = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of a metric."""
    name: str
    description: str
    data: list[MetricPoint] = field(default_factory=list)
    unit: str = ""
    
    def add(self, generation: int, value: float, metadata: Optional[dict] = None) -> None:
        """Add a data point."""
        self.data.append(MetricPoint(
            timestamp=time.time(),
            generation=generation,
            value=value,
            metadata=metadata or {},
        ))
    
    @property
    def latest(self) -> Optional[float]:
        """Get latest value."""
        return self.data[-1].value if self.data else None
    
    @property
    def average(self) -> float:
        """Get average value."""
        if not self.data:
            return 0.0
        return sum(p.value for p in self.data) / len(self.data)
    
    @property
    def trend(self) -> str:
        """Get trend direction."""
        if len(self.data) < 2:
            return "stable"
        
        recent = self.data[-5:] if len(self.data) >= 5 else self.data
        first = recent[0].value
        last = recent[-1].value
        
        if last > first * 1.05:
            return "increasing"
        elif last < first * 0.95:
            return "decreasing"
        return "stable"


class MetricsCollector:
    """
    Collects and manages evolution metrics.
    
    Tracks:
    - Best/average/worst fitness
    - Population diversity
    - Mutation rates
    - Selection pressure
    - Convergence indicators
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize metrics collector."""
        self.log_dir = log_dir or Path("metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric series
        self.metrics: dict[str, MetricSeries] = {}
        
        # Event log
        self.events: list[dict] = []
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self) -> None:
        """Initialize default metric series."""
        default_metrics = [
            ("best_fitness", "Best fitness in population", ""),
            ("avg_fitness", "Average fitness in population", ""),
            ("worst_fitness", "Worst fitness in population", ""),
            ("diversity", "Population diversity", ""),
            ("mutation_rate", "Actual mutation rate", ""),
            ("crossover_rate", "Crossover rate", ""),
            ("selection_pressure", "Selection pressure", ""),
            ("stagnation", "Stagnation counter", ""),
            ("convergence", "Convergence indicator", ""),
            ("unique_genomes", "Unique genome count", ""),
            ("population_size", "Population size", ""),
            ("execution_time", "Generation execution time", "seconds"),
        ]
        
        for name, desc, unit in default_metrics:
            self.metrics[name] = MetricSeries(name=name, description=desc, unit=unit)
    
    def record(self, generation: int, **kwargs) -> None:
        """Record metric values for a generation."""
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].add(generation, float(value))
            else:
                # Auto-create metric series
                self.metrics[name] = MetricSeries(name=name, description=name)
                self.metrics[name].add(generation, float(value))
    
    def log_event(self, event_type: str, data: dict) -> None:
        """Log an event."""
        self.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        })
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a specific metric series."""
        return self.metrics.get(name)
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        summary = {}
        for name, metric in self.metrics.items():
            summary[name] = {
                "latest": metric.latest,
                "average": metric.average,
                "trend": metric.trend,
                "data_points": len(metric.data),
            }
        return summary
    
    def get_fitness_progression(self) -> list[tuple[int, float]]:
        """Get best fitness progression over generations."""
        metric = self.metrics.get("best_fitness")
        if not metric:
            return []
        return [(p.generation, p.value) for p in metric.data]
    
    def get_diversity_progression(self) -> list[tuple[int, float]]:
        """Get diversity progression over generations."""
        metric = self.metrics.get("diversity")
        if not metric:
            return []
        return [(p.generation, p.value) for p in metric.data]
    
    def export_json(self, filepath: Optional[Path] = None) -> Path:
        """Export metrics to JSON."""
        filepath = filepath or self.log_dir / f"metrics_{int(time.time())}.json"
        
        data = {
            "metrics": {
                name: {
                    "description": m.description,
                    "unit": m.unit,
                    "data": [
                        {
                            "timestamp": p.timestamp,
                            "generation": p.generation,
                            "value": p.value,
                            "metadata": p.metadata,
                        }
                        for p in m.data
                    ],
                }
                for name, m in self.metrics.items()
            },
            "events": self.events,
            "summary": self.get_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def export_csv(self, metric_name: str, filepath: Optional[Path] = None) -> Optional[Path]:
        """Export a specific metric to CSV."""
        metric = self.metrics.get(metric_name)
        if not metric:
            return None
        
        filepath = filepath or self.log_dir / f"{metric_name}.csv"
        
        with open(filepath, 'w') as f:
            f.write("generation,timestamp,value\n")
            for point in metric.data:
                f.write(f"{point.generation},{point.timestamp},{point.value}\n")
        
        return filepath
    
    def plot(self, metric_name: str) -> str:
        """Generate ASCII plot of a metric."""
        metric = self.metrics.get(metric_name)
        if not metric or not metric.data:
            return "No data"
        
        values = [p.value for p in metric.data]
        generations = [p.generation for p in metric.data]
        
        if not values:
            return "No data"
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        height = 15
        width = min(len(values), 80)
        
        # Create buckets
        buckets = [[] for _ in range(height)]
        
        for i, v in enumerate(values[-width:]):
            bucket_idx = int((v - min_val) / range_val * (height - 1))
            bucket_idx = max(0, min(height - 1, bucket_idx))
            buckets[bucket_idx].append(i)
        
        # Build plot
        lines = []
        for row in range(height - 1, -1, -1):
            line = ""
            for col in range(width):
                if buckets[row] and col in buckets[row]:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        # Add labels
        result = [f"Metric: {metric_name}"]
        result.append("-" * width)
        result.extend(lines)
        result.append("-" * width)
        result.append(f"Gen: {generations[0]} to {generations[-1]}")
        result.append(f"Range: {min_val:.4f} to {max_val:.4f}")
        
        return "\n".join(result)
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        lines = ["Metrics Collector", "=" * 40]
        for name, data in summary.items():
            latest = data.get("latest", 0)
            trend = data.get("trend", "-")
            lines.append(f"  {name}: {latest:.4f} ({trend})")
        return "\n".join(lines)
