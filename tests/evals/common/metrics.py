"""
Métricas de evaluación profesionales para benchmarks de LLMs.

Este módulo proporciona métricas estándar y avanzadas para evaluar:
- MMLU: accuracy, confianza calibrada, consistencia
- GSM8K: accuracy numérica, pasos correctos, razonamiento válido
- RULER/Needle: recuperación, precisión posicional, memoria
- Métricas generales: latencia, uso de memoria, eficiencia
"""

import math
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

from .logging_utils import json_log


def accuracy(preds: List[int], targets: List[int]) -> float:
    """Calcula accuracy estándar."""
    if not preds:
        return 0.0
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    return correct / len(preds)


def accuracy_with_confidence(
    preds: List[int], 
    targets: List[int], 
    confidences: List[float]
) -> Dict[str, float]:
    """
    Calcula accuracy ponderado por confianza y métricas de calibración.
    
    Returns:
        Dict con accuracy, weighted_accuracy, ece (Expected Calibration Error)
    """
    if not preds:
        return {"accuracy": 0.0, "weighted_accuracy": 0.0, "ece": 0.0}
    
    # Accuracy estándar
    acc = accuracy(preds, targets)
    
    # Accuracy ponderada por confianza
    weighted_acc = sum(
        conf if p == t else (1 - conf) 
        for p, t, conf in zip(preds, targets, confidences)
    ) / len(preds)
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [(p, t, c) for p, t, c in zip(preds, targets, confidences) 
                  if bin_lower <= c < bin_upper]
        if in_bin:
            bin_acc = sum(1 for p, t, _ in in_bin if p == t) / len(in_bin)
            bin_conf = sum(c for _, _, c in in_bin) / len(in_bin)
            bin_weight = len(in_bin) / len(preds)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return {
        "accuracy": acc,
        "weighted_accuracy": weighted_acc,
        "ece": ece,
        "avg_confidence": np.mean(confidences) if confidences else 0.0
    }


def numeric_accuracy(
    pred_str: str, 
    target_str: str, 
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Evalúa accuracy numérica para GSM8K con tolerancia.
    
    Returns:
        Dict con exact_match, numeric_match, relative_error
    """
    def extract_number(s: str) -> Optional[float]:
        """Extrae número de string."""
        try:
            # Buscar números en el string
            import re
            numbers = re.findall(r'-?\d+\.?\d*', s.strip())
            if numbers:
                return float(numbers[-1])  # Tomar el último número
            return None
        except:
            return None
    
    pred_num = extract_number(pred_str)
    target_num = extract_number(target_str)
    
    exact_match = pred_str.strip() == target_str.strip()
    
    if pred_num is not None and target_num is not None:
        numeric_match = abs(pred_num - target_num) <= tolerance
        if target_num != 0:
            relative_error = abs(pred_num - target_num) / abs(target_num)
        else:
            relative_error = abs(pred_num - target_num)
    else:
        numeric_match = False
        relative_error = float('inf')
    
    return {
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "relative_error": relative_error,
        "pred_number": pred_num,
        "target_number": target_num
    }


def reasoning_quality(
    steps: List[str], 
    expected_answer: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evalúa calidad del razonamiento para GSM8K.
    
    Returns:
        Dict con step_count, avg_step_length, has_calculation, energy_consistency
    """
    if not steps:
        return {
            "step_count": 0,
            "avg_step_length": 0.0,
            "has_calculation": False,
            "energy_consistency": 0.0
        }
    
    # Contar pasos y longitud promedio
    step_count = len(steps)
    avg_step_length = np.mean([len(step.split()) for step in steps])
    
    # Detectar cálculos
    calc_keywords = ['+', '-', '*', '/', '=', 'sum', 'multiply', 'divide']
    has_calculation = any(
        any(keyword in step.lower() for keyword in calc_keywords)
        for step in steps
    )
    
    # Consistencia de energía (si está disponible en metadata)
    energy_consistency = 0.0
    if metadata and 'energy' in metadata:
        # Asumir que energía baja indica razonamiento más estable
        energy = metadata['energy']
        energy_consistency = max(0.0, 1.0 - min(energy / 100.0, 1.0))
    
    return {
        "step_count": step_count,
        "avg_step_length": avg_step_length,
        "has_calculation": has_calculation,
        "energy_consistency": energy_consistency
    }


def retrieval_metrics(
    retrieved: str, 
    expected: str, 
    position: Optional[int] = None,
    context_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Métricas de recuperación para RULER y Needle-in-Haystack.
    
    Returns:
        Dict con exact_match, contains_expected, position_accuracy, normalized_position
    """
    exact_match = retrieved.strip() == expected.strip()
    contains_expected = expected.strip() in retrieved.strip()
    
    # Métricas de posición (si están disponibles)
    position_accuracy = 0.0
    normalized_position = 0.0
    
    if position is not None and context_length is not None and context_length > 0:
        # Para needle-in-haystack, verificar si recuperó de la posición correcta
        expected_position_ratio = position / context_length
        normalized_position = expected_position_ratio
        
        # Asumir que si contiene el needle, la posición es correcta
        position_accuracy = 1.0 if contains_expected else 0.0
    
    return {
        "exact_match": exact_match,
        "contains_expected": contains_expected,
        "position_accuracy": position_accuracy,
        "normalized_position": normalized_position,
        "retrieved_length": len(retrieved),
        "expected_length": len(expected)
    }


def latency_metrics(times_ms: List[float]) -> Dict[str, float]:
    """Métricas de latencia."""
    if not times_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "std_ms": 0.0
        }
    
    times_array = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(times_array)),
        "median_ms": float(np.median(times_array)),
        "p95_ms": float(np.percentile(times_array, 95)),
        "p99_ms": float(np.percentile(times_array, 99)),
        "std_ms": float(np.std(times_array))
    }


def memory_usage_mb() -> float:
    """Uso de memoria en MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            _, peak = tracemalloc.get_traced_memory()
            return peak / (1024 * 1024)
        except Exception:
            return float("nan")


def perplexity(log_probs: List[float]) -> float:
    """Perplejidad desde log-probabilidades."""
    if not log_probs:
        return math.inf
    nll = -sum(log_probs) / len(log_probs)
    return math.exp(nll)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precisión, recall y F1."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r + 1e-12) if (p + r) > 0 else 0.0
    return p, r, f1


def confusion_counts(preds: List[int], targets: List[int], positive_label: int) -> Dict[str, int]:
    """Matriz de confusión."""
    tp = sum(int(p == positive_label and t == positive_label) for p, t in zip(preds, targets))
    fp = sum(int(p == positive_label and t != positive_label) for p, t in zip(preds, targets))
    fn = sum(int(p != positive_label and t == positive_label) for p, t in zip(preds, targets))
    tn = sum(int(p != positive_label and t != positive_label) for p, t in zip(preds, targets))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def consistency_score(predictions: List[int], confidences: List[float]) -> float:
    """
    Evalúa consistencia interna del modelo.
    Alta confianza con predicciones correctas = alta consistencia.
    """
    if not predictions or not confidences:
        return 0.0
    
    # Asumimos que no tenemos targets, solo evaluamos confianza vs consistencia
    # En producción, compararíamos con targets reales
    avg_confidence = np.mean(confidences)
    
    # Penalizar baja confianza general
    consistency = avg_confidence
    
    # Penalizar alta varianza en confianza
    if len(confidences) > 1:
        conf_std = np.std(confidences)
        consistency *= (1.0 - min(conf_std, 0.5))  # Penalización suave
    
    return float(consistency)


def benchmark_aggregation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrega resultados de múltiples ejecuciones.
    
    Returns:
        Dict con medias, desviaciones estándar y estadísticas agregadas
    """
    if not results:
        return {}
    
    aggregated = {}
    
    # Agrupar métricas por tipo
    metric_groups = defaultdict(list)
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                metric_groups[key].append(value)
    
    # Calcular estadísticas
    for metric_name, values in metric_groups.items():
        if values:
            values_array = np.array(values)
            aggregated[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "count": len(values)
            }
    
    return aggregated