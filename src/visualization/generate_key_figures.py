#!/usr/bin/env python3
"""Gera as principais visualizações finais a partir de artefatos persistidos.

Este módulo reproduz, fora do ambiente Jupyter, os gráficos finais de avaliação
e interpretabilidade consolidados nos notebooks de modelagem do projeto.

A implementação não recalcula o modelo, SHAP, Permutation Importance ou Partial
Dependence. Em vez disso, consome exclusivamente os artefatos persistidos nas
etapas anteriores, preservando a separação de responsabilidades adotada no
projeto:

- ``src/modeling`` produz o modelo e as predições;
- ``src/evaluation`` audita os resultados;
- ``src/visualization`` transforma artefatos validados em figuras reutilizáveis;
- os notebooks permanecem como registro completo da investigação e das decisões.

Por padrão, são geradas nove figuras em ``images/``:

1. matriz de confusão do teste temporal de 2024;
2. Permutation Importance — Top 15;
3. importância global SHAP — Top 15;
4. Partial Dependence — ``alunos_por_sala``;
5. Partial Dependence — ``pct_in_agua_rede_publica``;
6. Partial Dependence — ``pct_in_parque_infantil``;
7. risco previsto versus taxa observada de não alfabetização em 2024;
8. risco contextual versus desafio para a meta de 2025;
9. matriz de priorização municipal frente à meta de 2025.

Uso
---
A partir da raiz do repositório::

    python src/visualization/generate_key_figures.py

Os caminhos de entrada e saída também podem ser informados pela CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predicoes_modelo_final_2024.parquet"
)

DEFAULT_PERMUTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "importancia_permutation_modelo_final.parquet"
)

DEFAULT_SHAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "importancia_shap_modelo_final.parquet"
)

DEFAULT_PDP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "partial_dependence_modelo_final.parquet"
)

DEFAULT_MUNICIPAL_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validacao_municipal_2024.parquet"
)

DEFAULT_MUNICIPAL_PRIORITIZATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "priorizacao_municipal_meta_2025.parquet"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "images"

TARGET_COLUMN = "alfabetizado"
PREDICTION_COLUMN = "pred_nao_alfabetizado"

PDP_VARIABLES = (
    "alunos_por_sala",
    "pct_in_agua_rede_publica",
    "pct_in_parque_infantil",
)

TOP_N = 15
DEFAULT_DPI = 160

REQUIRED_PREDICTION_COLUMNS = {
    TARGET_COLUMN,
    PREDICTION_COLUMN,
}

REQUIRED_PERMUTATION_COLUMNS = {
    "variavel",
    "importancia_media",
    "desvio_padrao",
}

REQUIRED_SHAP_COLUMNS = {
    "variavel",
    "importancia_shap",
}

REQUIRED_PDP_COLUMNS = {
    "variavel",
    "valor_feature",
    "partial_dependence",
}

REQUIRED_MUNICIPAL_VALIDATION_COLUMNS = {
    "id_municipio",
    "id_municipio_nome",
    "risco_medio",
    "taxa_nao_alfabetizado_observada",
}

REQUIRED_MUNICIPAL_PRIORITIZATION_COLUMNS = {
    "id_municipio",
    "id_municipio_nome",
    "risco_medio",
    "desafio_meta_2025_pp",
    "priorizacao",
}


def parse_args() -> argparse.Namespace:
    """Lê os argumentos da linha de comando.

    Returns:
        Namespace com os caminhos dos artefatos de entrada, diretório de saída
        e resolução das figuras.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Gera as principais figuras finais a partir dos artefatos "
            "persistidos de avaliação e interpretabilidade."
        )
    )

    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=DEFAULT_PREDICTIONS_FILE,
        help=(
            "Predições do teste temporal. "
            f"Padrão: {DEFAULT_PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--permutation-file",
        type=Path,
        default=DEFAULT_PERMUTATION_FILE,
        help=(
            "Permutation Importance persistida. "
            f"Padrão: {DEFAULT_PERMUTATION_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--shap-file",
        type=Path,
        default=DEFAULT_SHAP_FILE,
        help=(
            "Importância global SHAP persistida. "
            f"Padrão: {DEFAULT_SHAP_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--pdp-file",
        type=Path,
        default=DEFAULT_PDP_FILE,
        help=(
            "Resultados de Partial Dependence persistidos. "
            f"Padrão: {DEFAULT_PDP_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--municipal-validation-file",
        type=Path,
        default=DEFAULT_MUNICIPAL_VALIDATION_FILE,
        help=(
            "Base persistida da validação municipal de 2024. "
            f"Padrão: {DEFAULT_MUNICIPAL_VALIDATION_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--municipal-prioritization-file",
        type=Path,
        default=DEFAULT_MUNICIPAL_PRIORITIZATION_FILE,
        help=(
            "Base persistida da priorização municipal frente à meta de 2025. "
            f"Padrão: {DEFAULT_MUNICIPAL_PRIORITIZATION_FILE.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Diretório de destino das figuras. "
            f"Padrão: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Resolução das imagens PNG. Padrão: {DEFAULT_DPI}.",
    )

    return parser.parse_args()


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    """Valida a presença das colunas necessárias em um artefato tabular.

    Args:
        dataframe: DataFrame carregado do artefato.
        required_columns: Colunas mínimas exigidas para a visualização.
        artifact_name: Nome legível do artefato usado na mensagem de erro.

    Raises:
        RuntimeError: Se o artefato estiver vazio ou faltar alguma coluna.
    """
    if dataframe.empty:
        raise RuntimeError(f"O artefato {artifact_name!r} está vazio.")

    missing_columns = set(required_columns) - set(dataframe.columns)

    if missing_columns:
        raise RuntimeError(
            f"Colunas obrigatórias ausentes em {artifact_name!r}: "
            f"{sorted(missing_columns)}"
        )


def load_parquet(
    input_file: Path,
    *,
    required_columns: Iterable[str],
    artifact_name: str,
) -> pd.DataFrame:
    """Carrega um parquet e valida o contrato mínimo esperado.

    Args:
        input_file: Caminho do arquivo parquet.
        required_columns: Colunas obrigatórias do artefato.
        artifact_name: Nome legível usado nas mensagens de validação.

    Returns:
        DataFrame validado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        RuntimeError: Se o conteúdo não atender ao contrato mínimo.
    """
    input_file = input_file.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado para {artifact_name}: {input_file}"
        )

    dataframe = pd.read_parquet(input_file)

    validate_columns(
        dataframe,
        required_columns,
        artifact_name=artifact_name,
    )

    return dataframe


def save_figure(
    figure: plt.Figure,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Salva uma figura e libera seus recursos.

    Args:
        figure: Figura Matplotlib já configurada.
        output_file: Caminho de destino da imagem.
        dpi: Resolução usada na exportação.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrix(
    predictions: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Gera a matriz de confusão do teste temporal.

    A classe ``Não`` é tratada como classe positiva, em consistência com a
    modelagem supervisionada consolidada no notebook 02.

    Args:
        predictions: Predições persistidas do teste temporal.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.

    Raises:
        RuntimeError: Se o target possuir rótulos diferentes de ``Sim`` e
            ``Não`` ou se as predições não forem binárias.
    """
    valid_labels = {"Sim", "Não"}
    observed_labels = set(predictions[TARGET_COLUMN].dropna().unique())

    unexpected_labels = observed_labels - valid_labels

    if unexpected_labels:
        raise RuntimeError(
            "Rótulos inesperados no target das predições: "
            f"{sorted(unexpected_labels)}"
        )

    if predictions[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            f"A coluna {TARGET_COLUMN!r} possui valores ausentes."
        )

    y_true = (
        predictions[TARGET_COLUMN]
        .map({"Sim": 0, "Não": 1})
        .astype("int8")
    )

    y_pred = predictions[PREDICTION_COLUMN]

    predicted_labels = set(y_pred.dropna().unique())

    if y_pred.isna().any() or not predicted_labels.issubset({0, 1}):
        raise RuntimeError(
            "As classificações persistidas devem conter apenas 0 e 1."
        )

    matrix = confusion_matrix(
        y_true,
        y_pred.astype("int8"),
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Alfabetizado", "Não alfabetizado"],
    )

    display.plot(
        ax=axis,
        values_format=",d",
    )

    axis.set_title("Matriz de confusão — teste temporal 2024")

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def plot_permutation_importance(
    importance: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
    top_n: int = TOP_N,
) -> None:
    """Gera o ranking das principais variáveis por Permutation Importance.

    Args:
        importance: Artefato com importância média e desvio-padrão.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.
        top_n: Número de características exibidas.
    """
    top_features = (
        importance
        .nlargest(top_n, "importancia_media")
        .sort_values("importancia_media")
    )

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.barh(
        top_features["variavel"],
        top_features["importancia_media"],
        xerr=top_features["desvio_padrao"],
    )

    axis.set_xlabel("Queda média em Average Precision")
    axis.set_ylabel("Variável")
    axis.set_title(
        "Permutation Importance — modelo final sem meta municipal"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def plot_shap_importance(
    importance: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
    top_n: int = TOP_N,
) -> None:
    """Gera o ranking global das principais características segundo SHAP.

    O artefato persistido contém a média dos valores SHAP absolutos calculada no
    notebook de modelagem. Por isso, esta função apenas reproduz a visualização
    global sem recalcular explicações SHAP.

    Args:
        importance: Artefato com a importância global SHAP.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.
        top_n: Número de características exibidas.
    """
    top_features = (
        importance
        .nlargest(top_n, "importancia_shap")
        .sort_values("importancia_shap")
    )

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.barh(
        top_features["variavel"],
        top_features["importancia_shap"],
    )

    axis.set_xlabel("Importância global — mean(|SHAP value|)")
    axis.set_ylabel("Característica")
    axis.set_title(
        "SHAP — importância global das características"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def plot_partial_dependence(
    partial_dependence: pd.DataFrame,
    variable: str,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Gera a curva de Partial Dependence de uma característica persistida.

    Args:
        partial_dependence: Artefato consolidado com os pontos das curvas PDP.
        variable: Característica que será exibida.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.

    Raises:
        RuntimeError: Se a característica solicitada não existir no artefato.
    """
    variable_data = (
        partial_dependence
        .loc[
            partial_dependence["variavel"] == variable,
            ["valor_feature", "partial_dependence"],
        ]
        .dropna()
        .sort_values("valor_feature")
    )

    if variable_data.empty:
        raise RuntimeError(
            "Não foram encontrados pontos de Partial Dependence para "
            f"{variable!r}."
        )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        variable_data["valor_feature"],
        variable_data["partial_dependence"],
    )

    axis.set_xlabel(variable)
    axis.set_ylabel("Partial Dependence")
    axis.set_title(f"Partial Dependence — {variable}")

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )



def plot_municipal_risk_validation(
    validation: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Gera a relação entre risco previsto e taxa observada em 2024.

    A figura reproduz a análise consolidada na seção 4.5 do notebook estratégico,
    utilizando o risco contextual médio previsto e a taxa municipal observada de
    não alfabetização.

    Args:
        validation: Base municipal persistida da validação temporal.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.
    """
    plot_data = validation[
        [
            "risco_medio",
            "taxa_nao_alfabetizado_observada",
        ]
    ].dropna()

    if plot_data.empty:
        raise RuntimeError(
            "A base de validação municipal não possui observações válidas."
        )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        plot_data["risco_medio"],
        plot_data["taxa_nao_alfabetizado_observada"],
        alpha=0.35,
    )

    axis.set_xlabel("Risco médio previsto")
    axis.set_ylabel("Taxa observada de não alfabetização")
    axis.set_title(
        "Risco contextual previsto × não alfabetização observada — 2024"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def plot_risk_vs_target_challenge(
    prioritization: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Gera a relação entre risco contextual e desafio para a meta de 2025.

    A figura reproduz a análise consolidada na seção 6.8 do notebook estratégico.

    Args:
        prioritization: Base municipal persistida da priorização frente à meta.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.
    """
    plot_data = prioritization[
        [
            "risco_medio",
            "desafio_meta_2025_pp",
        ]
    ].dropna()

    if plot_data.empty:
        raise RuntimeError(
            "A base de priorização não possui observações válidas "
            "para risco e desafio."
        )

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        plot_data["risco_medio"],
        plot_data["desafio_meta_2025_pp"],
        alpha=0.35,
    )

    axis.axhline(
        0,
        linestyle="--",
    )

    axis.set_xlabel("Risco contextual médio previsto")
    axis.set_ylabel("Avanço necessário para meta de 2025 (p.p.)")
    axis.set_title(
        "Risco contextual × desafio para a meta de 2025"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def plot_municipal_prioritization_matrix(
    prioritization: pd.DataFrame,
    output_file: Path,
    *,
    dpi: int,
) -> None:
    """Gera a matriz de priorização municipal frente à meta de 2025.

    As linhas de corte são recalculadas a partir das medianas das duas dimensões
    persistidas, reproduzindo a regra utilizada no notebook estratégico.

    Args:
        prioritization: Base municipal com risco, desafio e grupo de priorização.
        output_file: Caminho do PNG de saída.
        dpi: Resolução usada na exportação.

    Raises:
        RuntimeError: Se houver grupos de priorização inesperados ou dados
            insuficientes para calcular as medianas.
    """
    expected_groups = {
        "Alto risco + alto desafio",
        "Alto risco + baixo desafio",
        "Baixo risco + alto desafio",
        "Baixo risco + baixo desafio",
    }

    observed_groups = set(
        prioritization["priorizacao"].dropna().unique()
    )

    unexpected_groups = observed_groups - expected_groups

    if unexpected_groups:
        raise RuntimeError(
            "Grupos de priorização inesperados: "
            f"{sorted(unexpected_groups)}"
        )

    plot_data = prioritization[
        [
            "risco_medio",
            "desafio_meta_2025_pp",
            "priorizacao",
        ]
    ].dropna()

    if plot_data.empty:
        raise RuntimeError(
            "A base de priorização não possui observações válidas."
        )

    median_risk = plot_data["risco_medio"].median()
    median_challenge = plot_data["desafio_meta_2025_pp"].median()

    figure, axis = plt.subplots(figsize=(9, 7))

    for group, group_data in plot_data.groupby("priorizacao"):
        axis.scatter(
            group_data["risco_medio"],
            group_data["desafio_meta_2025_pp"],
            alpha=0.35,
            label=group,
        )

    axis.axvline(
        median_risk,
        linestyle="--",
    )

    axis.axhline(
        median_challenge,
        linestyle="--",
    )

    axis.set_xlabel("Risco contextual médio previsto")
    axis.set_ylabel("Avanço necessário para meta de 2025 (p.p.)")
    axis.set_title("Matriz de priorização municipal — meta 2025")
    axis.legend()

    figure.tight_layout()

    save_figure(
        figure,
        output_file,
        dpi=dpi,
    )


def generate_key_figures(
    predictions: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    partial_dependence: pd.DataFrame,
    municipal_validation: pd.DataFrame,
    municipal_prioritization: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Gera todas as figuras-chave definidas para a primeira versão.

    Args:
        predictions: Predições persistidas do teste temporal.
        permutation_importance: Resultado persistido da Permutation Importance.
        shap_importance: Importância global SHAP persistida.
        partial_dependence: Pontos persistidos das curvas de Partial Dependence.
        municipal_validation: Base persistida da validação municipal de 2024.
        municipal_prioritization: Base persistida da priorização municipal frente
            à meta de 2025.
        output_dir: Diretório de destino das figuras.
        dpi: Resolução usada na exportação.

    Returns:
        Lista dos caminhos das imagens geradas.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    confusion_file = output_dir / "confusion_matrix_2024.png"
    plot_confusion_matrix(
        predictions,
        confusion_file,
        dpi=dpi,
    )
    output_files.append(confusion_file)

    permutation_file = output_dir / "permutation_importance_top15.png"
    plot_permutation_importance(
        permutation_importance,
        permutation_file,
        dpi=dpi,
    )
    output_files.append(permutation_file)

    shap_file = output_dir / "shap_importance_top15.png"
    plot_shap_importance(
        shap_importance,
        shap_file,
        dpi=dpi,
    )
    output_files.append(shap_file)

    for variable in PDP_VARIABLES:
        pdp_file = output_dir / f"partial_dependence_{variable}.png"

        plot_partial_dependence(
            partial_dependence,
            variable,
            pdp_file,
            dpi=dpi,
        )

        output_files.append(pdp_file)

    municipal_validation_file = (
        output_dir / "risco_previsto_vs_observado_2024.png"
    )
    plot_municipal_risk_validation(
        municipal_validation,
        municipal_validation_file,
        dpi=dpi,
    )
    output_files.append(municipal_validation_file)

    risk_target_file = (
        output_dir / "risco_contextual_vs_desafio_meta_2025.png"
    )
    plot_risk_vs_target_challenge(
        municipal_prioritization,
        risk_target_file,
        dpi=dpi,
    )
    output_files.append(risk_target_file)

    prioritization_matrix_file = (
        output_dir / "matriz_priorizacao_municipal_2025.png"
    )
    plot_municipal_prioritization_matrix(
        municipal_prioritization,
        prioritization_matrix_file,
        dpi=dpi,
    )
    output_files.append(prioritization_matrix_file)

    return output_files


def main() -> None:
    """Carrega os artefatos persistidos e gera as figuras-chave."""
    args = parse_args()

    predictions = load_parquet(
        args.predictions_file,
        required_columns=REQUIRED_PREDICTION_COLUMNS,
        artifact_name="predições do teste temporal",
    )

    permutation_importance = load_parquet(
        args.permutation_file,
        required_columns=REQUIRED_PERMUTATION_COLUMNS,
        artifact_name="Permutation Importance",
    )

    shap_importance = load_parquet(
        args.shap_file,
        required_columns=REQUIRED_SHAP_COLUMNS,
        artifact_name="importância global SHAP",
    )

    partial_dependence = load_parquet(
        args.pdp_file,
        required_columns=REQUIRED_PDP_COLUMNS,
        artifact_name="Partial Dependence",
    )

    municipal_validation = load_parquet(
        args.municipal_validation_file,
        required_columns=REQUIRED_MUNICIPAL_VALIDATION_COLUMNS,
        artifact_name="validação municipal de 2024",
    )

    municipal_prioritization = load_parquet(
        args.municipal_prioritization_file,
        required_columns=REQUIRED_MUNICIPAL_PRIORITIZATION_COLUMNS,
        artifact_name="priorização municipal frente à meta de 2025",
    )

    generated_files = generate_key_figures(
        predictions,
        permutation_importance,
        shap_importance,
        partial_dependence,
        municipal_validation,
        municipal_prioritization,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print("Figuras geradas com sucesso:")

    for output_file in generated_files:
        try:
            display_path = output_file.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = output_file

        print(f"  - {display_path}")


if __name__ == "__main__":
    main()
