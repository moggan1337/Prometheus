"""
Evaluator Module - Measures genome fitness through tests and benchmarks.

The Evaluator is responsible for:
1. Running test suites against genomes
2. Benchmarking performance characteristics
3. Measuring code quality metrics
4. Calculating multi-objective fitness scores
"""

from __future__ import annotations
import ast
import time
import tracemalloc
import statistics
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .genome import Genome, FitnessScores, GenomeType


@dataclass
class EvaluationResult:
    """Detailed results from genome evaluation."""
    genome: Genome
    test_results: dict
    benchmark_results: dict
    quality_metrics: dict
    execution_time: float
    memory_peak: float
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and self.test_results.get("passed", 0) > 0


@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    code: str
    expected_output: Any = None
    timeout: float = 5.0
    setup_code: Optional[str] = None


class Evaluator:
    """
    Evaluates genome fitness through comprehensive testing and benchmarking.
    
    The Evaluator runs multiple evaluation strategies:
    - Unit testing: Correctness validation
    - Performance benchmarking: Speed and efficiency
    - Static analysis: Code quality metrics
    - Memory profiling: Resource usage
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize evaluator with optional configuration."""
        self.config = config or self._default_config()
        self.timeout = self.config.get("timeout", 30)
        self.max_memory_mb = self.config.get("max_memory_mb", 512)
        self.warmup_runs = self.config.get("warmup_runs", 3)
        self.benchmark_runs = self.config.get("benchmark_runs", 10)
        
    def _default_config(self) -> dict:
        """Get default evaluator configuration."""
        return {
            "timeout": 30,
            "max_memory_mb": 512,
            "warmup_runs": 3,
            "benchmark_runs": 10,
            "correctness_weight": 0.3,
            "performance_weight": 0.2,
            "complexity_weight": 0.1,
            "maintainability_weight": 0.15,
            "memory_weight": 0.1,
            "readability_weight": 0.15,
        }
    
    def evaluate(
        self, 
        genome: Genome, 
        test_suite: Optional[list[TestCase]] = None
    ) -> FitnessScores:
        """
        Comprehensive evaluation of a genome.
        
        Args:
            genome: The genome to evaluate
            test_suite: Optional list of test cases
            
        Returns:
            FitnessScores object with multi-objective fitness values
        """
        fitness = FitnessScores()
        
        # Run test suite
        if test_suite:
            test_results = self._run_tests(genome, test_suite)
            fitness.correctness = test_results.get("pass_rate", 0.0)
        else:
            # Generate basic test cases
            generated_tests = self._generate_basic_tests(genome)
            test_results = self._run_tests(genome, generated_tests)
            fitness.correctness = test_results.get("pass_rate", 0.0)
        
        # Benchmark performance
        benchmark_results = self._benchmark(genome)
        fitness.performance = benchmark_results.get("score", 0.5)
        
        # Analyze complexity
        complexity_metrics = self._analyze_complexity(genome)
        fitness.complexity = complexity_metrics.get("normalized", 0.5)
        
        # Measure maintainability
        maintainability = self._measure_maintainability(genome)
        fitness.maintainability = maintainability
        
        # Profile memory usage
        memory_usage = self._profile_memory(genome)
        fitness.memory = memory_usage
        
        # Evaluate readability
        readability = self._measure_readability(genome)
        fitness.readability = readability
        
        # Calculate composite score
        fitness.update_composite(self.config)
        
        return fitness
    
    def _run_tests(
        self, 
        genome: Genome, 
        test_suite: list[TestCase]
    ) -> dict:
        """Execute test suite against genome code."""
        results = {
            "total": len(test_suite),
            "passed": 0,
            "failed": 0,
            "errors": [],
            "pass_rate": 0.0,
            "execution_time": 0.0,
        }
        
        if not test_suite:
            return results
        
        start_time = time.time()
        
        for test in test_suite:
            try:
                test_result = self._execute_test(genome, test)
                if test_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(test_result.get("error"))
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))
        
        results["execution_time"] = time.time() - start_time
        results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
        
        return results
    
    def _execute_test(self, genome: Genome, test: TestCase) -> dict:
        """Execute a single test case."""
        result = {"passed": False, "output": None, "error": None}
        
        # Build execution context
        setup = test.setup_code or ""
        full_code = f"{setup}\n{genome.source}\n{test.code}"
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.py', 
                delete=False
            ) as f:
                f.write(full_code)
                temp_path = f.name
            
            # Execute with timeout
            exec_globals = {"__name__": "__test__"}
            exec_globals.update(self.config.get("builtins", {}))
            
            # Run the test
            exec(compile(full_code, temp_path, 'exec'), exec_globals)
            
            result["passed"] = True
            
        except subprocess.TimeoutExpired:
            result["error"] = "Test timeout"
        except Exception as e:
            result["error"] = str(e)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return result
    
    def _benchmark(self, genome: Genome) -> dict:
        """
        Benchmark genome performance.
        
        Returns:
            Dictionary with timing statistics
        """
        benchmark_code = f"""
import time
import statistics

{genome.source}

# Benchmark harness
def run_benchmark():
    times = []
    for _ in range({self.benchmark_runs}):
        start = time.perf_counter()
        # Execute with test input
        try:
            result = solve() if 'solve' in dir() else main() if 'main' in dir() else None
        except:
            result = None
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {{
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }}

result = run_benchmark()
"""
        
        try:
            exec_globals = {}
            exec(compile(benchmark_code, "<benchmark>", 'exec'), exec_globals)
            benchmark_data = exec_globals.get("result", {})
            
            # Calculate performance score (inverse of time, normalized)
            mean_time = benchmark_data.get("mean", 1.0)
            # Score: faster = higher (max 1.0)
            score = max(0.0, min(1.0, 1.0 / (mean_time * 1000 + 1)))
            
            return {**benchmark_data, "score": score}
            
        except Exception as e:
            return {"error": str(e), "score": 0.0}
    
    def _analyze_complexity(self, genome: Genome) -> dict:
        """
        Analyze code complexity using AST.
        
        Returns:
            Dictionary with complexity metrics
        """
        metrics = {
            "cyclomatic": 0,
            "cognitive": 0,
            "lines": 0,
            "parameters": 0,
            "returns": 0,
            "conditionals": 0,
            "loops": 0,
            "nested_depth": 0,
            "normalized": 0.5,
        }
        
        try:
            tree = ast.parse(genome.source)
            metrics["lines"] = genome.line_count
            
            # Calculate cyclomatic complexity
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    metrics["cyclomatic"] += 1
                elif isinstance(node, ast.BoolOp):
                    metrics["cyclomatic"] += len(node.values) - 1
                    
            # Count function parameters
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["parameters"] = max(metrics["parameters"], len(node.args.args))
                    
            # Count returns, conditionals, loops
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    metrics["returns"] += 1
                if isinstance(node, (ast.If, ast.IfExp)):
                    metrics["conditionals"] += 1
                if isinstance(node, (ast.While, ast.For)):
                    metrics["loops"] += 1
            
            # Calculate nested depth
            metrics["nested_depth"] = self._calculate_nesting_depth(tree)
            
            # Cognitive complexity approximation
            metrics["cognitive"] = (
                metrics["cyclomatic"] + 
                metrics["conditionals"] * 0.5 + 
                metrics["loops"] * 0.7 +
                metrics["nested_depth"] * 0.3
            )
            
            # Normalize complexity (0 = simple, 1 = complex)
            # Based on typical ranges
            ideal_lines = 50
            ideal_cyclomatic = 10
            complexity_score = (
                min(1.0, metrics["lines"] / ideal_lines) * 0.3 +
                min(1.0, metrics["cyclomatic"] / ideal_cyclomatic) * 0.4 +
                min(1.0, metrics["nested_depth"] / 5) * 0.3
            )
            metrics["normalized"] = min(1.0, complexity_score)
            
        except SyntaxError:
            metrics["normalized"] = 1.0  # Max complexity for invalid code
        
        return metrics
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth of AST."""
        max_depth = [0]
        
        def visit(node, depth=0):
            max_depth[0] = max(max_depth[0], depth)
            for child in ast.iter_child_nodes(node):
                visit(child, depth + 1)
        
        visit(tree)
        return max_depth[0]
    
    def _measure_maintainability(self, genome: Genome) -> float:
        """
        Measure code maintainability.
        
        Returns:
            Score from 0.0 (unmaintainable) to 1.0 (highly maintainable)
        """
        score = 1.0
        
        # Penalize for various issues
        issues = []
        
        # Check for long lines
        for i, line in enumerate(genome.source.splitlines(), 1):
            if len(line) > 120:
                score -= 0.02
                issues.append(f"Line {i} exceeds 120 chars")
        
        # Check for missing docstrings
        try:
            tree = ast.parse(genome.source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        score -= 0.05
                        issues.append(f"Missing docstring: {node.name}")
        except:
            score -= 0.1
        
        # Check for meaningful variable names
        try:
            tree = ast.parse(genome.source)
            short_names = 0
            total_names = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    total_names += 1
                    if len(node.id) <= 2 and node.id not in ('i', 'j', 'k', 'x', 'y', 'z'):
                        short_names += 1
            
            if total_names > 0 and short_names / total_names > 0.3:
                score -= 0.1
        except:
            pass
        
        # Check for type hints
        if "->" not in genome.source and ":" not in genome.source.split("def ")[1] if "def " in genome.source else True:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _profile_memory(self, genome: Genome) -> float:
        """
        Profile memory usage of genome.
        
        Returns:
            Normalized memory score (0 = high usage, 1 = low usage)
        """
        profile_code = f"""
import tracemalloc
import gc

{genome.source}

# Memory profiling harness
def profile_memory():
    gc.collect()
    tracemalloc.start()
    
    # Run the code
    try:
        result = solve() if 'solve' in dir() else main() if 'main' in dir() else None
    except:
        result = None
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {{
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
    }}

result = profile_memory()
"""
        
        try:
            exec_globals = {}
            exec(compile(profile_code, "<memory_profile>", 'exec'), exec_globals)
            memory_data = exec_globals.get("result", {})
            peak_mb = memory_data.get("peak_mb", 100)
            
            # Normalize: lower memory = higher score
            # Assume 100MB is average, scale accordingly
            return max(0.0, min(1.0, 1.0 - (peak_mb / 100)))
            
        except Exception:
            return 0.5  # Default if profiling fails
    
    def _measure_readability(self, genome: Genome) -> float:
        """
        Measure code readability.
        
        Returns:
            Score from 0.0 (unreadable) to 1.0 (highly readable)
        """
        score = 1.0
        
        lines = genome.source.splitlines()
        
        # Check blank lines (too few or too many)
        blank_ratio = sum(1 for l in lines if not l.strip()) / max(1, len(lines))
        if blank_ratio < 0.05:
            score -= 0.1  # Too cramped
        elif blank_ratio > 0.3:
            score -= 0.05  # Too sparse
        
        # Check indentation consistency
        indentations = []
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(stripped)
                if indent % 4 != 0:
                    score -= 0.05
                indentations.append(indent)
        
        # Check for comments
        comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        comment_ratio = comment_lines / max(1, len(lines))
        if comment_ratio < 0.05:
            score -= 0.1
        elif comment_ratio > 0.4:
            score -= 0.05
        
        # Check naming conventions
        try:
            import re
            snake_case = len(re.findall(r'\b[a-z][a-z0-9_]*\b', genome.source))
            total_names = len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', genome.source))
            if total_names > 0 and snake_case / total_names < 0.5:
                score -= 0.1
        except:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _generate_basic_tests(self, genome: Genome) -> list[TestCase]:
        """Generate basic test cases for a genome."""
        tests = []
        
        # Basic execution test
        tests.append(TestCase(
            name="basic_execution",
            code="try:\n    result = solve() if 'solve' in dir() else main() if 'main' in dir() else None\nexcept Exception as e:\n    raise AssertionError(f'Execution failed: {{e}}')",
            expected_output=None,
        ))
        
        # Import test
        tests.append(TestCase(
            name="import_test",
            code="try:\n    import sys\n    __import__('sys')\nexcept Exception as e:\n    raise AssertionError(f'Import failed: {{e}}')",
        ))
        
        return tests
    
    def evaluate_batch(self, genomes: list[Genome]) -> list[FitnessScores]:
        """Evaluate multiple genomes in parallel."""
        with ThreadPoolExecutor(max_workers=min(len(genomes), 4)) as executor:
            futures = [executor.submit(self.evaluate, g) for g in genomes]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=self.timeout))
                except FuturesTimeoutError:
                    results.append(FitnessScores())
                except Exception:
                    results.append(FitnessScores())
            return results


# Convenience function
def quick_evaluate(source: str) -> FitnessScores:
    """Quickly evaluate a piece of code."""
    genome = Genome(source=source)
    evaluator = Evaluator()
    return evaluator.evaluate(genome)
