#!/usr/bin/env python3
"""Avalia e audita as predições persistidas do modelo final.

Este módulo implementa uma avaliação independente dos artefatos produzidos por
``src/modeling/train_final_model.py``. Ele não realiza treinamento nem modifica o
modelo persistido.

A avaliação segue as decisões consolidadas nos notebooks de modelagem do projeto:
a classe positiva é ``Não`` (aluno não alfabetizado) e as métricas principais do
teste temporal de 2024 são ROC-AUC, Average Precision, Balanced Accuracy,
Precision, Recall e F1.

Além de recalcular as métricas diretamente a partir das predições persistidas, o
módulo audita a integridade das probabilidades, das classes previstas, a aplicação
do threshold definido em 2023 e a consistência das métricas com o metadata
produzido no retreino final.

Uso
---
A partir da raiz do repositório::

    python src/evaluation/metrics.py

O comando termina com erro caso alguma validação de integridade ou consistência
não seja satisfeita.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "predicoes_modelo_final_2024.parquet"
)
DEFAULT_METADATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "treino_modelo_final_metadata.json"
)

TARGET_COLUMN = "alfabetizado"
PROBABILITY_COLUMN = "prob_nao_alfabetizado"
PREDICTION_COLUMN = "pred_nao_alfabetizado"
POSITIVE_LABEL = "Não"
NEGATIVE_LABEL = "Sim"
VALID_TARGET_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}

REQUIRED_PREDICTION_COLUMNS = {
    TARGET_COLUMN,
    PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
}

METRIC_NAMES = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)

METRIC_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando.

    Returns:
        Namespace com os caminhos dos artefatos que serão avaliados.
    """
    parser = argparse.ArgumentParser(
        description="Recalcula e audita as métricas do teste temporal do modelo final."
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=DEFAULT_PREDICTIONS_FILE,
        help=(
            "Arquivo parquet com as predições do teste temporal. "
            f"Padrão: {DEFAULT_PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help=(
            "Metadata JSON produzido pelo treino final. "
            f"Padrão: {DEFAULT_METADATA_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    return parser.parse_args()


def load_predictions(input_file: Path) -> pd.DataFrame:
    """Carrega e valida estruturalmente o artefato de predições.

    Args:
        input_file: Caminho do parquet de predições persistidas.

    Returns:
        DataFrame com as predições do teste temporal.

    Raises:
        FileNotFoundError: Se o arquivo de predições não existir.
        RuntimeError: Se colunas obrigatórias estiverem ausentes ou se o artefato
            não possuir registros.
    """
    input_file = input_file.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo de predições não encontrado: {input_file}")

    predictions = pd.read_parquet(input_file)
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)

    if missing_columns:
        raise RuntimeError(
            "Colunas obrigatórias ausentes no artefato de predições: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:
        raise RuntimeError("O artefato de predições está vazio.")

    return predictions


def load_metadata(input_file: Path) -> dict[str, Any]:
    """Carrega o metadata gerado pelo retreino final.

    Args:
        input_file: Caminho do arquivo JSON de metadata.

    Returns:
        Dicionário com as decisões e resultados persistidos no treinamento.

    Raises:
        FileNotFoundError: Se o arquivo de metadata não existir.
        RuntimeError: Se blocos obrigatórios estiverem ausentes.
    """
    input_file = input_file.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo de metadata não encontrado: {input_file}")

    with input_file.open(encoding="utf-8") as file:
        metadata = json.load(file)

    required_blocks = {
        "temporal_test_year",
        "positive_class",
        "threshold_selection",
        "temporal_test_metrics",
        "temporal_test_rows",
    }
    missing_blocks = required_blocks - set(metadata)

    if missing_blocks:
        raise RuntimeError(
            "Blocos obrigatórios ausentes no metadata: "
            f"{sorted(missing_blocks)}"
        )

    return metadata


def build_binary_target(
    predictions: pd.DataFrame,
    *,
    positive_label: str = POSITIVE_LABEL,
) -> np.ndarray:
    """Converte o target textual para a representação binária do projeto.

    Args:
        predictions: DataFrame com a coluna textual ``alfabetizado``.
        positive_label: Rótulo textual que representa a classe positiva.

    Returns:
        Vetor binário com ``1`` para a classe positiva e ``0`` para a negativa.

    Raises:
        RuntimeError: Se o target possuir valores ausentes ou rótulos fora do
            contrato esperado do projeto.
    """
    target = predictions[TARGET_COLUMN]

    if target.isna().any():
        raise RuntimeError(f"A coluna {TARGET_COLUMN!r} possui valores ausentes.")

    unexpected_labels = set(target.unique()) - VALID_TARGET_LABELS
    if unexpected_labels:
        raise RuntimeError(
            f"Valores inesperados na coluna {TARGET_COLUMN!r}: "
            f"{sorted(unexpected_labels)}; esperado {sorted(VALID_TARGET_LABELS)}."
        )

    return target.eq(positive_label).astype(np.int8).to_numpy()


def validate_prediction_artifact(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Audita probabilidades, classes previstas e aplicação do threshold.

    Args:
        predictions: Artefato de predições persistidas.
        threshold: Limiar de decisão selecionado com previsões OOF de 2023.

    Returns:
        Resumo das validações realizadas.

    Raises:
        RuntimeError: Se forem encontrados nulos, probabilidades inválidas,
            classes diferentes de 0/1 ou inconsistência com o threshold.
    """
    probabilities = predictions[PROBABILITY_COLUMN].to_numpy(dtype=float)
    predicted_classes = predictions[PREDICTION_COLUMN].to_numpy()

    if not np.isfinite(probabilities).all():
        raise RuntimeError("As probabilidades possuem valores ausentes ou não finitos.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise RuntimeError("Foram encontradas probabilidades fora do intervalo [0, 1].")

    if pd.isna(predicted_classes).any():
        raise RuntimeError("A coluna de classificações previstas possui valores ausentes.")

    unique_classes = set(np.unique(predicted_classes).tolist())

    if not unique_classes.issubset({0, 1}):
        raise RuntimeError(
            "As classificações previstas devem pertencer a {0, 1}; "
            f"encontrado: {sorted(unique_classes)}"
        )

    expected_classes = (probabilities >= threshold).astype(np.int8)
    actual_classes = predicted_classes.astype(np.int8)

    if not np.array_equal(actual_classes, expected_classes):
        mismatches = int((actual_classes != expected_classes).sum())
        raise RuntimeError(
            "As classificações persistidas não reproduzem o threshold "
            f"registrado no metadata. Divergências: {mismatches:,}."
        )

    counts = pd.Series(actual_classes).value_counts().sort_index()

    return {
        "rows": int(len(predictions)),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "predicted_class_counts": {
            str(int(label)): int(count)
            for label, count in counts.items()
        },
        "threshold_consistent": True,
    }


def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calcula as métricas finais utilizadas na avaliação temporal.

    Args:
        y_true: Classes verdadeiras binárias, com ``1`` como classe positiva.
        probabilities: Probabilidades estimadas para a classe positiva.
        predictions: Classes previstas após aplicação do threshold.

    Returns:
        Dicionário com ROC-AUC, Average Precision, Balanced Accuracy, Precision,
        Recall e F1.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
    }


def compare_metrics_with_metadata(
    calculated_metrics: Mapping[str, float],
    metadata_metrics: Mapping[str, float],
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> dict[str, dict[str, Any]]:
    """Compara métricas recalculadas com as persistidas pelo treinamento.

    Args:
        calculated_metrics: Métricas recalculadas a partir das predições.
        metadata_metrics: Métricas registradas no metadata do treino.
        tolerance: Tolerância absoluta usada na comparação numérica.

    Returns:
        Resultado detalhado da comparação por métrica.

    Raises:
        RuntimeError: Se alguma métrica estiver ausente no metadata ou se a
            diferença ultrapassar a tolerância definida.
    """
    comparison: dict[str, dict[str, Any]] = {}
    inconsistent_metrics: list[str] = []

    for metric_name in METRIC_NAMES:
        if metric_name not in metadata_metrics:
            raise RuntimeError(f"Métrica {metric_name!r} ausente no metadata.")

        calculated_value = float(calculated_metrics[metric_name])
        metadata_value = float(metadata_metrics[metric_name])

        consistent = bool(
            np.isclose(
                calculated_value,
                metadata_value,
                rtol=0.0,
                atol=tolerance,
            )
        )

        comparison[metric_name] = {
            "calculated": calculated_value,
            "metadata": metadata_value,
            "absolute_difference": abs(calculated_value - metadata_value),
            "consistent": consistent,
        }

        if not consistent:
            inconsistent_metrics.append(metric_name)

    if inconsistent_metrics:
        raise RuntimeError(
            "As métricas recalculadas divergem do metadata: "
            f"{inconsistent_metrics}"
        )

    return comparison


def evaluate_persisted_predictions(
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Executa a avaliação independente do teste temporal persistido.

    Args:
        predictions: Predições produzidas pelo modelo final no teste temporal.
        metadata: Metadata gerado pelo script de treinamento.

    Returns:
        Relatório com integridade do artefato, métricas recalculadas e
        consistência em relação ao metadata.

    Raises:
        RuntimeError: Se a classe positiva do metadata não corresponder à
            definição adotada no projeto ou se qualquer auditoria falhar.
    """
    positive_class = metadata["positive_class"]

    if positive_class != POSITIVE_LABEL:
        raise RuntimeError(
            "Classe positiva inesperada no metadata: "
            f"{positive_class!r}; esperado {POSITIVE_LABEL!r}."
        )

    expected_rows = int(metadata["temporal_test_rows"])
    if len(predictions) != expected_rows:
        raise RuntimeError(
            "Quantidade de linhas do artefato de predições diverge do metadata: "
            f"encontrado={len(predictions):,}, esperado={expected_rows:,}."
        )

    threshold = float(metadata["threshold_selection"]["threshold"])

    artifact_validation = validate_prediction_artifact(
        predictions,
        threshold=threshold,
    )

    y_true = build_binary_target(
        predictions,
        positive_label=positive_class,
    )

    probabilities = predictions[PROBABILITY_COLUMN].to_numpy(dtype=float)
    predicted_classes = predictions[PREDICTION_COLUMN].to_numpy(dtype=np.int8)

    metrics = calculate_binary_metrics(
        y_true,
        probabilities,
        predicted_classes,
    )

    metric_comparison = compare_metrics_with_metadata(
        metrics,
        metadata["temporal_test_metrics"],
    )

    return {
        "temporal_test_year": int(metadata["temporal_test_year"]),
        "temporal_test_rows": expected_rows,
        "positive_class": positive_class,
        "threshold": threshold,
        "artifact_validation": artifact_validation,
        "metrics": metrics,
        "metadata_comparison": metric_comparison,
    }


def print_evaluation_report(report: Mapping[str, Any]) -> None:
    """Exibe no console um resumo legível da avaliação independente.

    Args:
        report: Relatório produzido por ``evaluate_persisted_predictions``.
    """
    validation = report["artifact_validation"]
    metrics = report["metrics"]

    print(f"Teste temporal: {report['temporal_test_year']}")
    print(f"Classe positiva: {report['positive_class']}")
    print(f"Registros avaliados: {validation['rows']:,}")
    print(f"Registros esperados (metadata): {report['temporal_test_rows']:,}")
    print(f"Threshold auditado: {report['threshold']:.10f}")

    print("\nIntegridade do artefato:")
    print(
        "  Probabilidades: "
        f"{validation['probability_min']:.6f} "
        f"a {validation['probability_max']:.6f}"
    )
    print(
        "  Classes previstas: "
        f"{validation['predicted_class_counts']}"
    )
    print("  Aplicação do threshold: OK")

    print("\nMétricas recalculadas:")
    for metric_name in METRIC_NAMES:
        print(f"  {metric_name}: {metrics[metric_name]:.6f}")

    print("\nValidação contra metadata:")
    for metric_name in METRIC_NAMES:
        comparison = report["metadata_comparison"][metric_name]
        status = "OK" if comparison["consistent"] else "DIVERGENTE"
        print(f"  {metric_name}: {status}")

    print("\nAvaliação independente validada com sucesso.")


def main() -> None:
    """Executa a auditoria dos artefatos persistidos do modelo final."""
    args = parse_args()

    predictions_file = args.predictions_file.expanduser().resolve()
    metadata_file = args.metadata_file.expanduser().resolve()

    print(f"Carregando predições: {predictions_file}")
    predictions = load_predictions(predictions_file)

    print(f"Carregando metadata: {metadata_file}")
    metadata = load_metadata(metadata_file)

    report = evaluate_persisted_predictions(predictions, metadata)

    print()
    print_evaluation_report(report)


if __name__ == "__main__":
    main()
