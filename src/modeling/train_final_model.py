#!/usr/bin/env python3
"""Retreina e materializa o modelo final de risco de não alfabetização.

Este módulo operacionaliza a configuração final definida no notebook
``02_preparacao_modelagem.ipynb`` do Tech Challenge - Fase 3. O objetivo é
permitir o retreino completo do modelo sem depender de uma sessão Jupyter,
preservando as decisões metodológicas consolidadas no desenvolvimento:

- população de desenvolvimento em 2023 e teste temporal final em 2024;
- classe positiva igual a ``Não alfabetizado``;
- remoção de ``meta_alfabetizacao_2025`` após o teste de ablação;
- conjunto final de 36 preditores: ``rede`` + 35 características do Censo;
- imputação numérica pela mediana e categórica pela moda;
- ``OneHotEncoder`` para ``rede``, com categorias desconhecidas ignoradas;
- ``HistGradientBoostingClassifier`` dentro de um ``Pipeline``;
- ``StratifiedGroupKFold`` com agrupamento por escola;
- otimização por ``RandomizedSearchCV`` usando Average Precision;
- definição do threshold pela maximização do F1 em probabilidades OOF de 2023;
- avaliação temporal preservada: 2024 não participa de seleção, otimização ou
  definição do limiar;
- cálculo explícito das métricas finais no teste temporal de 2024;
- persistência local do pipeline treinado com o threshold operacional embutido;
- persistência das predições de 2024 e dos metadados de rastreabilidade.

O notebook permanece como registro analítico e narrativo do projeto. Este
script representa a implementação reproduzível da solução final já escolhida,
e não uma nova etapa de experimentação.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Project paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.thresholded_model import ThresholdedBinaryClassifier

DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = REPO_ROOT / "model"

DEFAULT_INPUT_FILE = PROCESSED_DIR / "alunos_modelagem_enriquecido.parquet"
DEFAULT_PREDICTIONS_FILE = PROCESSED_DIR / "predicoes_modelo_final_2024.parquet"
DEFAULT_METADATA_FILE = PROCESSED_DIR / "treino_modelo_final_metadata.json"
DEFAULT_MODEL_FILE = MODEL_DIR / "modelo_final.joblib"


# ============================================================
# Final modeling contract
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
SEARCH_N_ITER = 10
SEARCH_N_JOBS = -1
OOF_N_JOBS = 2

DEV_YEAR = 2023
TEST_YEAR = 2024
TARGET = "alfabetizado"
GROUP = "id_escola"
POSITIVE_LABEL = "Não"
NEGATIVE_LABEL = "Sim"

EXPECTED_ROWS = {
    DEV_YEAR: 1_502_809,
    TEST_YEAR: 1_851_852,
}

# Conjunto final definido após o teste de ablação da meta municipal.
# A meta_alfabetizacao_2025 não participa do modelo definitivo.
FINAL_FEATURES = [
    "rede",
    "qtd_escolas",
    "pct_tp_localizacao_1",
    "pct_localizacao_diferenciada",
    "pct_in_agua_potavel",
    "pct_in_agua_rede_publica",
    "pct_in_energia_rede_publica",
    "pct_in_esgoto_rede_publica",
    "pct_in_lixo_servico_coleta",
    "pct_in_banheiro_pne",
    "pct_in_biblioteca",
    "pct_in_biblioteca_sala_leitura",
    "pct_in_laboratorio_ciencias",
    "pct_in_laboratorio_informatica",
    "pct_in_patio_coberto",
    "pct_in_patio_descoberto",
    "pct_in_parque_infantil",
    "pct_in_quadra_esportes",
    "pct_in_acessibilidade_corrimao",
    "pct_in_acessibilidade_rampas",
    "pct_in_acessibilidade_sinal_visual",
    "pct_in_computador",
    "pct_in_internet",
    "pct_in_internet_aprendizagem",
    "pct_in_banda_larga",
    "media_qt_salas_utilizadas",
    "media_qt_desktop_aluno",
    "total_qt_desktop_aluno",
    "media_qt_comp_portatil_aluno",
    "total_qt_comp_portatil_aluno",
    "media_qt_tablet_aluno",
    "media_qt_mat_bas",
    "media_qt_doc_bas",
    "alunos_por_docente",
    "alunos_por_sala",
    "dispositivos_aluno_por_100_matriculas",
]

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_iter": [100, 200],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 1.0, 5.0],
}

PREDICTION_ID_COLUMNS = [
    "ano",
    "id_municipio",
    "id_municipio_nome",
    "id_escola",
    "id_aluno",
    "alfabetizado",
]


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando.

    Returns:
        argparse.Namespace: Argumentos validados pela interface de linha de
        comando.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Retreina o modelo final sem meta municipal, avalia o teste "
            "temporal e persiste o artefato treinado."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=(
            "Base enriquecida de modelagem. Padrão: "
            "data/processed/alunos_modelagem_enriquecido.parquet"
        ),
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=DEFAULT_PREDICTIONS_FILE,
        help=(
            "Destino das predições de 2024. Padrão: "
            "data/processed/predicoes_modelo_final_2024.parquet"
        ),
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=(
            "Destino do modelo final treinado. Padrão: "
            "model/modelo_final.joblib"
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help=(
            "Destino dos metadados do retreino. Padrão: "
            "data/processed/treino_modelo_final_metadata.json"
        ),
    )
    return parser.parse_args()


# ============================================================
# Data contract
# ============================================================


def load_modeling_data(input_file: Path) -> pd.DataFrame:
    """Carrega e valida a base enriquecida usada pelo modelo final.

    Args:
        input_file: Caminho do Parquet enriquecido produzido pela etapa de
            preprocessing.

    Returns:
        pd.DataFrame: Base validada contendo desenvolvimento de 2023 e teste
        temporal de 2024.

    Raises:
        FileNotFoundError: Se a base de entrada não existir.
        RuntimeError: Se faltarem colunas obrigatórias, houver anos
        inesperados, divergência nas quantidades esperadas ou valores inválidos
        na variável-alvo.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Base de modelagem não encontrada: {input_file}")

    print(f"Carregando base: {input_file}")
    df = pd.read_parquet(input_file)

    required_columns = {
        "ano",
        TARGET,
        GROUP,
        *PREDICTION_ID_COLUMNS,
        *FINAL_FEATURES,
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Colunas obrigatórias ausentes na base enriquecida: "
            f"{sorted(missing_columns)}"
        )

    counts = df["ano"].value_counts().to_dict()
    unexpected_years = set(counts) - set(EXPECTED_ROWS)

    if unexpected_years:
        raise RuntimeError(
            f"Anos inesperados na base de modelagem: {sorted(unexpected_years)}"
        )

    for year, expected_rows in EXPECTED_ROWS.items():
        observed_rows = counts.get(year, 0)
        if observed_rows != expected_rows:
            raise RuntimeError(
                f"{year}: {observed_rows:,} registros encontrados; "
                f"esperado {expected_rows:,}."
            )

    target_values = set(df[TARGET].dropna().unique())
    expected_target_values = {POSITIVE_LABEL, NEGATIVE_LABEL}

    if target_values != expected_target_values:
        raise RuntimeError(
            "Valores inesperados na variável-alvo: "
            f"{sorted(target_values)}. Esperado: {sorted(expected_target_values)}."
        )

    if df[TARGET].isna().any():
        raise RuntimeError("A variável-alvo contém valores ausentes.")

    if df[GROUP].isna().any():
        raise RuntimeError("A variável de agrupamento id_escola contém nulos.")

    print(f"Linhas carregadas: {len(df):,}")
    print(f"Features finais: {len(FINAL_FEATURES)}")

    return df


def prepare_temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Constrói desenvolvimento, teste temporal, alvo e grupos por escola.

    A classe positiva é ``Não alfabetizado`` (1). O ano de 2023 é usado
    exclusivamente para desenvolvimento e 2024 é preservado como teste temporal
    final.

    Args:
        df: Base enriquecida validada.

    Returns:
        tuple: ``X_dev``, ``X_test``, ``y_dev``, ``y_test`` e ``groups_dev``.
    """
    mask_dev = df["ano"] == DEV_YEAR
    mask_test = df["ano"] == TEST_YEAR

    X_dev = df.loc[mask_dev, FINAL_FEATURES].copy()
    X_test = df.loc[mask_test, FINAL_FEATURES].copy()

    target_mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

    y_dev = df.loc[mask_dev, TARGET].map(target_mapping).astype("int8")
    y_test = df.loc[mask_test, TARGET].map(target_mapping).astype("int8")
    groups_dev = df.loc[mask_dev, GROUP].copy()

    print(f"Desenvolvimento {DEV_YEAR}: {len(X_dev):,}")
    print(f"Teste temporal {TEST_YEAR}: {len(X_test):,}")
    print(f"Escolas no desenvolvimento: {groups_dev.nunique():,}")

    return X_dev, X_test, y_dev, y_test, groups_dev


def diagnose_temporal_categorical_drift(
    X_dev: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Diagnostica categorias presentes no teste e ausentes no desenvolvimento.

    A verificação formaliza situações esperadas em validação temporal nas quais
    uma categoria pode surgir somente no período futuro. O diagnóstico não
    altera os dados nem o pipeline: o ``OneHotEncoder(handle_unknown="ignore")``
    continua responsável por tratar categorias desconhecidas durante a
    transformação.

    Args:
        X_dev: Preditores do conjunto de desenvolvimento de 2023.
        X_test: Preditores do teste temporal de 2024.

    Returns:
        dict[str, Any]: Bloco de auditoria contendo, para cada variável
        categórica, categorias observadas em desenvolvimento e teste,
        categorias inéditas no teste, quantidade e proporção de linhas
        afetadas, estratégia de tratamento e status do diagnóstico.
    """
    categorical_features = [
        column
        for column in FINAL_FEATURES
        if not pd.api.types.is_numeric_dtype(X_dev[column])
    ]

    diagnostics: dict[str, Any] = {}

    for feature in categorical_features:
        development_categories = sorted(
            str(value)
            for value in X_dev[feature].dropna().unique().tolist()
        )
        temporal_test_categories = sorted(
            str(value)
            for value in X_test[feature].dropna().unique().tolist()
        )

        development_set = set(development_categories)
        temporal_test_set = set(temporal_test_categories)
        unknown_categories = sorted(temporal_test_set - development_set)

        if unknown_categories:
            unknown_mask = X_test[feature].astype("string").isin(
                unknown_categories
            )
            affected_rows = int(unknown_mask.sum())
            unknown_category_counts = {
                str(category): int(count)
                for category, count in (
                    X_test.loc[unknown_mask, feature]
                    .astype("string")
                    .value_counts(dropna=False)
                    .items()
                )
            }
            status = "expected_behavior"
        else:
            affected_rows = 0
            unknown_category_counts = {}
            status = "no_unknown_categories"

        diagnostics[feature] = {
            "development_categories": development_categories,
            "temporal_test_categories": temporal_test_categories,
            "unknown_categories": unknown_categories,
            "unknown_category_counts": unknown_category_counts,
            "affected_rows": affected_rows,
            "affected_share": float(affected_rows / len(X_test)),
            "handling": (
                "OneHotEncoder(handle_unknown='ignore') encodes unseen "
                "categories as all zeros in the learned dummy columns"
            ),
            "status": status,
        }

    print("Diagnóstico de categorias no teste temporal:")
    for feature, result in diagnostics.items():
        if result["unknown_categories"]:
            print(
                f"  {feature}: categorias novas em {TEST_YEAR} = "
                f"{result['unknown_categories']} | "
                f"linhas afetadas = {result['affected_rows']:,} "
                f"({result['affected_share']:.6%})"
            )
        else:
            print(
                f"  {feature}: nenhuma categoria inédita identificada em "
                f"{TEST_YEAR}."
            )

    return {
        "unknown_categories_temporal_test": diagnostics,
    }


# ============================================================
# Final pipeline
# ============================================================


def build_final_pipeline(X_dev: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline do HistGradientBoosting sem meta municipal.

    O pipeline reproduz o preprocessing definido no notebook final: variáveis
    numéricas são imputadas pela mediana; a variável categórica ``rede`` é
    imputada pela categoria mais frequente e codificada por One-Hot Encoding.
    Não é aplicada padronização às variáveis numéricas, pois o estimador final é
    baseado em árvores.

    Args:
        X_dev: Preditores do conjunto de desenvolvimento usados para inferir os
            tipos numéricos e categóricos.

    Returns:
        Pipeline: Pipeline não ajustado contendo ``ColumnTransformer`` e
        ``HistGradientBoostingClassifier``.

    Raises:
        RuntimeError: Se a separação de tipos não reproduzir o contrato
        esperado de uma feature categórica e 35 features numéricas.
    """
    categorical_features = [
        column
        for column in FINAL_FEATURES
        if not pd.api.types.is_numeric_dtype(X_dev[column])
    ]
    numeric_features = [
        column
        for column in FINAL_FEATURES
        if pd.api.types.is_numeric_dtype(X_dev[column])
    ]

    if categorical_features != ["rede"]:
        raise RuntimeError(
            "Contrato categórico inesperado. "
            f"Encontrado: {categorical_features}; esperado: ['rede']."
        )

    if len(numeric_features) != 35:
        raise RuntimeError(
            f"Quantidade inesperada de features numéricas: {len(numeric_features)}; "
            "esperado 35."
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    estimator = HistGradientBoostingClassifier(
        max_leaf_nodes=31,
        max_iter=100,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def build_cross_validation() -> StratifiedGroupKFold:
    """Cria a estratégia de validação estratificada e agrupada por escola.

    Returns:
        StratifiedGroupKFold: Validador com cinco folds, embaralhamento e seed
        fixa, conforme a configuração final do notebook.
    """
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def validate_group_separation(
    cv: StratifiedGroupKFold,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    groups_dev: pd.Series,
) -> None:
    """Valida que nenhuma escola aparece em treino e validação no mesmo fold.

    Args:
        cv: Estratégia de validação cruzada.
        X_dev: Preditores de desenvolvimento.
        y_dev: Alvo binário de desenvolvimento.
        groups_dev: Identificador de escola usado como grupo.

    Raises:
        RuntimeError: Se algum fold compartilhar escolas entre treino e
        validação.
    """
    for fold, (train_idx, validation_idx) in enumerate(
        cv.split(X_dev, y_dev, groups=groups_dev),
        start=1,
    ):
        train_schools = set(groups_dev.iloc[train_idx])
        validation_schools = set(groups_dev.iloc[validation_idx])
        overlap = train_schools.intersection(validation_schools)

        if overlap:
            raise RuntimeError(
                f"Fold {fold}: {len(overlap)} escolas aparecem em treino e validação."
            )

    print("Separação por escola validada nos cinco folds.")


# ============================================================
# Optimization and threshold
# ============================================================


def optimize_model(
    pipeline: Pipeline,
    cv: StratifiedGroupKFold,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    groups_dev: pd.Series,
) -> RandomizedSearchCV:
    """Reotimiza o modelo final exclusivamente sobre os dados de 2023.

    A busca reproduz o espaço de hiperparâmetros adotado após o teste de ablação
    e usa Average Precision como função objetivo.

    Args:
        pipeline: Pipeline base do modelo sem meta municipal.
        cv: Estratégia de validação cruzada agrupada por escola.
        X_dev: Preditores de desenvolvimento de 2023.
        y_dev: Alvo binário de desenvolvimento.
        groups_dev: Identificador de escola para os grupos da validação.

    Returns:
        RandomizedSearchCV: Busca ajustada, incluindo o melhor estimador.
    """
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=SEARCH_N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        verbose=1,
        refit=True,
    )

    search.fit(
        X_dev,
        y_dev,
        groups=groups_dev,
    )

    print(f"Melhor Average Precision CV: {search.best_score_:.6f}")
    print("Melhores hiperparâmetros:")
    for parameter, value in sorted(search.best_params_.items()):
        print(f"  {parameter}: {value}")

    return search


def select_oof_threshold(
    estimator: Pipeline,
    cv: StratifiedGroupKFold,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    groups_dev: pd.Series,
) -> dict[str, float]:
    """Seleciona o limiar que maximiza o F1 em probabilidades OOF de 2023.

    Cada probabilidade é produzida por um modelo que não utilizou aquela
    observação durante o ajuste. O conjunto temporal de 2024 não participa da
    seleção do limiar.

    Args:
        estimator: Melhor pipeline encontrado na otimização.
        cv: Estratégia de validação cruzada agrupada por escola.
        X_dev: Preditores de desenvolvimento.
        y_dev: Alvo binário de desenvolvimento.
        groups_dev: Identificador de escola usado nos folds.

    Returns:
        dict[str, float]: Threshold selecionado e Precision, Recall e F1 no ponto
        ótimo das previsões OOF.
    """
    probabilities_oof = cross_val_predict(
        estimator,
        X_dev,
        y_dev,
        groups=groups_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=OOF_N_JOBS,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_dev,
        probabilities_oof,
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1))
    threshold = float(thresholds[best_index])

    result = {
        "threshold": threshold,
        "precision_oof": float(precision[best_index]),
        "recall_oof": float(recall[best_index]),
        "f1_oof": float(f1[best_index]),
    }

    print(f"Threshold selecionado: {threshold:.4f}")
    print(f"Precision OOF: {result['precision_oof']:.4f}")
    print(f"Recall OOF: {result['recall_oof']:.4f}")
    print(f"F1 OOF: {result['f1_oof']:.4f}")

    return result


# ============================================================
# Final fit and persistence
# ============================================================


def fit_and_predict_temporal_test(
    estimator: Pipeline,
    threshold: float,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Ajusta o modelo em todo 2023 e produz as predições temporais de 2024.

    Args:
        estimator: Pipeline selecionado após a otimização.
        threshold: Limiar definido exclusivamente pelas previsões OOF de 2023.
        X_dev: Preditores de desenvolvimento de 2023.
        y_dev: Alvo binário de desenvolvimento.
        X_test: Preditores do teste temporal de 2024.

    Returns:
        tuple: Pipeline final ajustado, probabilidades da classe positiva em
        2024 e classificações binárias calculadas pelo threshold selecionado.
    """
    final_model = estimator
    final_model.fit(X_dev, y_dev)

    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype("int8")

    print(f"Observações previstas em {TEST_YEAR}: {len(probabilities):,}")

    return final_model, probabilities, predictions



def evaluate_temporal_test(
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calcula as métricas finais no teste temporal de 2024.

    A avaliação é realizada somente após todas as decisões de modelagem,
    hiperparâmetros e threshold terem sido definidas com dados de 2023.

    Args:
        y_test: Alvo binário observado no teste temporal.
        probabilities: Probabilidades previstas para a classe positiva.
        predictions: Classificações binárias produzidas pelo threshold OOF.

    Returns:
        dict[str, float]: ROC-AUC, Average Precision, Balanced Accuracy,
        Precision, Recall e F1 no teste temporal.

    Raises:
        RuntimeError: Se os vetores de entrada tiverem dimensões
        incompatíveis.
    """
    if len(y_test) != len(probabilities) or len(y_test) != len(predictions):
        raise RuntimeError(
            "Dimensões incompatíveis na avaliação temporal: "
            f"y_test={len(y_test):,}, probabilidades={len(probabilities):,}, "
            f"predições={len(predictions):,}."
        )

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }

    print(f"Métricas do teste temporal {TEST_YEAR}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def wrap_final_model(
    estimator: Pipeline,
    threshold: float,
) -> ThresholdedBinaryClassifier:
    """Envolve o pipeline treinado com o threshold operacional selecionado.

    Args:
        estimator: Pipeline final já ajustado em todo o desenvolvimento de
            2023.
        threshold: Limiar selecionado por maximização do F1 em previsões OOF.

    Returns:
        ThresholdedBinaryClassifier: Artefato final com pré-processamento,
        estimador probabilístico e regra de decisão.
    """
    return ThresholdedBinaryClassifier(
        estimator=estimator,
        threshold=threshold,
    )


def validate_persisted_model_contract(
    model: ThresholdedBinaryClassifier,
    X_test: pd.DataFrame,
    expected_probabilities: np.ndarray,
    expected_predictions: np.ndarray,
) -> None:
    """Valida que o wrapper reproduz probabilidades e classes já calculadas.

    Args:
        model: Modelo final com threshold embutido.
        X_test: Preditores do teste temporal.
        expected_probabilities: Probabilidades produzidas pelo pipeline bruto.
        expected_predictions: Classes obtidas pela regra ``>= threshold``.

    Raises:
        RuntimeError: Se probabilidades ou classificações divergirem das
        saídas originais.
    """
    model_probabilities = model.predict_proba(X_test)[:, 1]
    model_predictions = model.predict(X_test)

    if not np.allclose(
        model_probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "O modelo envolvido alterou as probabilidades do pipeline final."
        )

    if not np.array_equal(model_predictions, expected_predictions):
        raise RuntimeError(
            "O modelo envolvido não reproduz as classificações do threshold "
            "selecionado."
        )

    print("Sanidade do modelo final validada no teste temporal.")


def save_model(
    model: ThresholdedBinaryClassifier,
    output_file: Path,
) -> None:
    """Persiste localmente o modelo final com threshold embutido.

    Args:
        model: Artefato final contendo pipeline treinado e threshold.
        output_file: Caminho de destino do arquivo Joblib.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)
    print(f"Modelo final salvo em: {output_file}")


def build_predictions_artifact(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Monta o artefato de predições consumido pela aplicação estratégica.

    Args:
        df: Base enriquecida completa com identificadores e resultado observado.
        probabilities: Probabilidades estimadas de não alfabetização em 2024.
        predictions: Classificações binárias obtidas com o threshold OOF.

    Returns:
        pd.DataFrame: Predições de 2024 com identificadores territoriais e
        escolares, resultado observado, probabilidade e classe prevista.

    Raises:
        RuntimeError: Se a quantidade de probabilidades ou classificações não
        coincidir com a quantidade de registros do teste temporal.
    """
    mask_test = df["ano"] == TEST_YEAR
    artifact = df.loc[mask_test, PREDICTION_ID_COLUMNS].copy()

    if len(artifact) != len(probabilities) or len(artifact) != len(predictions):
        raise RuntimeError(
            "Dimensões incompatíveis entre o teste temporal e as predições: "
            f"linhas={len(artifact):,}, probabilidades={len(probabilities):,}, "
            f"predições={len(predictions):,}."
        )

    artifact["prob_nao_alfabetizado"] = probabilities
    artifact["pred_nao_alfabetizado"] = predictions

    return artifact


def save_predictions(artifact: pd.DataFrame, output_file: Path) -> None:
    """Persiste as predições temporais em Parquet.

    Args:
        artifact: DataFrame produzido por ``build_predictions_artifact``.
        output_file: Caminho de destino do arquivo Parquet.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(output_file, index=False)

    print(f"Predições salvas em: {output_file}")
    print(f"Linhas persistidas: {len(artifact):,}")


def save_training_metadata(
    output_file: Path,
    search: RandomizedSearchCV,
    threshold_result: dict[str, float],
    temporal_test_metrics: dict[str, float],
    temporal_test_rows: int,
    data_diagnostics: dict[str, Any],
    model_file: Path,
) -> None:
    """Persiste os parâmetros e resultados necessários à rastreabilidade.

    Args:
        output_file: Caminho do arquivo JSON de metadados.
        search: Busca de hiperparâmetros já ajustada.
        threshold_result: Resultado da seleção do threshold OOF.
        temporal_test_metrics: Métricas finais calculadas em 2024.
        temporal_test_rows: Quantidade de observações do teste temporal de 2024 utilizadas na avaliação, usada para auditoria externa.
        data_diagnostics: Diagnósticos de diferenças categóricas entre o
            desenvolvimento e o teste temporal.
        model_file: Caminho do artefato Joblib persistido localmente.
    """
    try:
        model_file_metadata = str(model_file.relative_to(REPO_ROOT))
    except ValueError:
        model_file_metadata = str(model_file)

    metadata: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "development_year": DEV_YEAR,
        "temporal_test_year": TEST_YEAR,
        "positive_class": POSITIVE_LABEL,
        "group_column": GROUP,
        "cv": {
            "strategy": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
        },
        "optimization": {
            "strategy": "RandomizedSearchCV",
            "n_iter": SEARCH_N_ITER,
            "scoring": "average_precision",
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "parameter_space": PARAM_DISTRIBUTIONS,
        },
        "threshold_selection": {
            "criterion": "maximize_f1_oof_2023",
            **threshold_result,
            "application": (
                "embedded_in_persisted_model_predict: positive class when "
                "predict_proba[:, 1] >= threshold"
            ),
        },
        "temporal_test_metrics": temporal_test_metrics,
        "temporal_test_rows": temporal_test_rows,
        "data_diagnostics": data_diagnostics,
        "model_artifact": {
            "file": model_file_metadata,
            "format": "joblib",
            "persisted": True,
            "contains_preprocessing": True,
            "contains_threshold": True,
            "wrapper_class": "src.modeling.thresholded_model.ThresholdedBinaryClassifier",
        },
        "features": FINAL_FEATURES,
        "excluded_after_ablation": "meta_alfabetizacao_2025",
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Metadados salvos em: {output_file}")


# ============================================================
# Orchestration
# ============================================================


def train_final_model(
    input_file: Path,
    predictions_file: Path,
    metadata_file: Path,
    model_file: Path,
) -> None:
    """Executa o fluxo reproduzível de retreino do modelo final.

    Args:
        input_file: Base enriquecida de modelagem.
        predictions_file: Destino das predições temporais de 2024.
        metadata_file: Destino do JSON com parâmetros, métricas e threshold.
        model_file: Destino local do modelo final persistido em Joblib.
    """
    df = load_modeling_data(input_file)

    X_dev, X_test, y_dev, y_test, groups_dev = prepare_temporal_split(df)

    data_diagnostics = diagnose_temporal_categorical_drift(
        X_dev=X_dev,
        X_test=X_test,
    )

    cv = build_cross_validation()
    validate_group_separation(cv, X_dev, y_dev, groups_dev)

    pipeline = build_final_pipeline(X_dev)
    search = optimize_model(
        pipeline=pipeline,
        cv=cv,
        X_dev=X_dev,
        y_dev=y_dev,
        groups_dev=groups_dev,
    )

    threshold_result = select_oof_threshold(
        estimator=search.best_estimator_,
        cv=cv,
        X_dev=X_dev,
        y_dev=y_dev,
        groups_dev=groups_dev,
    )

    (
        fitted_pipeline,
        probabilities_test,
        predictions_test,
    ) = fit_and_predict_temporal_test(
        estimator=search.best_estimator_,
        threshold=threshold_result["threshold"],
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
    )

    temporal_test_metrics = evaluate_temporal_test(
        y_test=y_test,
        probabilities=probabilities_test,
        predictions=predictions_test,
    )

    final_model = wrap_final_model(
        estimator=fitted_pipeline,
        threshold=threshold_result["threshold"],
    )
    validate_persisted_model_contract(
        model=final_model,
        X_test=X_test,
        expected_probabilities=probabilities_test,
        expected_predictions=predictions_test,
    )

    artifact = build_predictions_artifact(
        df=df,
        probabilities=probabilities_test,
        predictions=predictions_test,
    )

    save_predictions(artifact, predictions_file)
    save_model(final_model, model_file)
    save_training_metadata(
        output_file=metadata_file,
        search=search,
        threshold_result=threshold_result,
        temporal_test_metrics=temporal_test_metrics,
        temporal_test_rows=len(y_test),
        data_diagnostics=data_diagnostics,
        model_file=model_file,
    )

    print("Retreino concluído com sucesso.")


def main() -> None:
    """Executa o retreino a partir dos argumentos de linha de comando."""
    args = parse_args()
    train_final_model(
        input_file=args.input_file,
        predictions_file=args.predictions_file,
        metadata_file=args.metadata_file,
        model_file=args.model_file,
    )


if __name__ == "__main__":
    main()
