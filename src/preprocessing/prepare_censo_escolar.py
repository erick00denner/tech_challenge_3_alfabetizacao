#!/usr/bin/env python3
"""Prepara características municipais do Censo Escolar para o Tech Challenge - Fase 3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar"
DEFAULT_YEARS = (2022, 2023)
DEPENDENCIA_MAP = {"1": "Federal", "2": "Estadual", "3": "Municipal", "4": "Privada"}

SITUACAO_FUNCIONAMENTO_COLUMN = "TP_SITUACAO_FUNCIONAMENTO"
SITUACAO_FUNCIONAMENTO_ALLOWED = {"1", "2", "3"}
SITUACAO_EM_ATIVIDADE = "1"

KEY_COLUMNS = ["CO_MUNICIPIO", "TP_DEPENDENCIA"]
CATEGORICAL_COLUMNS = {"TP_LOCALIZACAO": ["1", "2"]}
LOCALIZACAO_DIFERENCIADA_COLUMN = "TP_LOCALIZACAO_DIFERENCIADA"
LOCALIZACAO_DIFERENCIADA_ALLOWED = {"0", "1", "2", "3", "8"}

BINARY_COLUMNS = [
    "IN_AGUA_POTAVEL", "IN_AGUA_REDE_PUBLICA", "IN_ENERGIA_REDE_PUBLICA",
    "IN_ESGOTO_REDE_PUBLICA", "IN_LIXO_SERVICO_COLETA", "IN_BANHEIRO_PNE",
    "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_CIENCIAS",
    "IN_LABORATORIO_INFORMATICA", "IN_PATIO_COBERTO", "IN_PATIO_DESCOBERTO",
    "IN_PARQUE_INFANTIL", "IN_QUADRA_ESPORTES", "IN_ACESSIBILIDADE_CORRIMAO",
    "IN_ACESSIBILIDADE_RAMPAS", "IN_ACESSIBILIDADE_SINAL_VISUAL", "IN_COMPUTADOR",
    "IN_INTERNET", "IN_INTERNET_APRENDIZAGEM", "IN_BANDA_LARGA",
]

QUANTITATIVE_COLUMNS = [
    "QT_SALAS_UTILIZADAS", "QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO",
    "QT_TABLET_ALUNO", "QT_MAT_BAS", "QT_DOC_BAS",
]

SELECTED_COLUMNS = (
    [SITUACAO_FUNCIONAMENTO_COLUMN]
    + KEY_COLUMNS
    + list(CATEGORICAL_COLUMNS)
    + [LOCALIZACAO_DIFERENCIADA_COLUMN]
    + BINARY_COLUMNS
    + QUANTITATIVE_COLUMNS
)

DETAILED_COLUMNS = [LOCALIZACAO_DIFERENCIADA_COLUMN] + BINARY_COLUMNS + QUANTITATIVE_COLUMNS
CHUNK_SIZE = 25_000


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepara features municipais do Censo Escolar.")
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "external")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def _add_frames(current: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Acumula estatísticas parciais dos chunks."""
    return new.copy() if current is None else current.add(new, fill_value=0)


def _find_main_csv(zf: ZipFile, year: int) -> str:
    """Localiza o CSV principal no ZIP oficial."""
    matches = [n for n in zf.namelist() if n.lower().endswith(f"microdados_ed_basica_{year}.csv")]
    if len(matches) != 1:
        raise RuntimeError(f"{year}: esperado exatamente 1 CSV principal, encontrados {len(matches)}: {matches}")
    return matches[0]


def _validate_header(zip_path: Path, year: int) -> str:
    """Valida as colunas obrigatórias do arquivo oficial."""
    with ZipFile(zip_path) as zf:
        csv_name = _find_main_csv(zf, year)
        with zf.open(csv_name) as raw:
            header = pd.read_csv(raw, sep=";", encoding="latin1", nrows=0).columns.tolist()
    missing = sorted(set(SELECTED_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(f"{year}: colunas obrigatórias ausentes: {missing}")
    return csv_name


def _normalize_strings(chunk: pd.DataFrame) -> pd.DataFrame:
    """Padroniza strings e converte vazios em nulos."""
    for col in chunk.columns:
        chunk[col] = chunk[col].astype("string").str.strip().replace("", pd.NA)
    return chunk


def _validate_domains(chunk: pd.DataFrame, year: int) -> None:
    """Valida domínios categóricos usados no processamento."""
    situacao = set(chunk[SITUACAO_FUNCIONAMENTO_COLUMN].dropna().unique()) - SITUACAO_FUNCIONAMENTO_ALLOWED
    if situacao:
        raise RuntimeError(
            f"{year}: {SITUACAO_FUNCIONAMENTO_COLUMN} contém valores inesperados: {sorted(situacao)}"
        )

    dep = set(chunk["TP_DEPENDENCIA"].dropna().unique()) - set(DEPENDENCIA_MAP)
    if dep:
        raise RuntimeError(f"{year}: TP_DEPENDENCIA contém valores inesperados: {sorted(dep)}")

    for col, allowed in CATEGORICAL_COLUMNS.items():
        unexpected = set(chunk[col].dropna().unique()) - set(allowed)
        if unexpected:
            raise RuntimeError(f"{year}: {col} contém valores inesperados: {sorted(unexpected)}")

    unexpected_loc_dif = (
        set(chunk[LOCALIZACAO_DIFERENCIADA_COLUMN].dropna().unique())
        - LOCALIZACAO_DIFERENCIADA_ALLOWED
    )
    if unexpected_loc_dif:
        raise RuntimeError(
            f"{year}: {LOCALIZACAO_DIFERENCIADA_COLUMN} contém valores inesperados: {sorted(unexpected_loc_dif)}"
        )

    for col in BINARY_COLUMNS:
        unexpected = set(chunk[col].dropna().unique()) - {"0", "1"}
        if unexpected:
            raise RuntimeError(f"{year}: {col} contém valores inesperados: {sorted(unexpected)}")


def _aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Agrega um chunk de escolas ativas por município e dependência."""
    chunk = chunk.dropna(subset=KEY_COLUMNS).copy()

    for col in BINARY_COLUMNS + QUANTITATIVE_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk["_INFO_DETALHADA_COMPLETA"] = chunk[DETAILED_COLUMNS].notna().all(axis=1).astype("int8")
    grouped = chunk.groupby(KEY_COLUMNS, dropna=False, observed=True)
    parts = []

    base = pd.DataFrame(index=grouped.size().index)
    base["_qtd_escolas"] = grouped.size()
    base["_info_completa_sum"] = grouped["_INFO_DETALHADA_COMPLETA"].sum()
    parts.append(base)

    for col, categories in CATEGORICAL_COLUMNS.items():
        parts.append(grouped[col].count().rename(f"_{col}_valid").to_frame())
        for category in categories:
            temp = chunk[KEY_COLUMNS].copy()
            temp["_indicator"] = (chunk[col] == category).fillna(False).astype("int8")
            counts = temp.groupby(KEY_COLUMNS, observed=True)["_indicator"].sum().rename(f"_{col}_{category}_sum")
            parts.append(counts.to_frame())

    loc_col = LOCALIZACAO_DIFERENCIADA_COLUMN
    parts.append(grouped[loc_col].count().rename(f"_{loc_col}_valid").to_frame())
    temp = chunk[KEY_COLUMNS].copy()
    temp["_indicator"] = chunk[loc_col].isin(["1", "2", "3", "8"]).fillna(False).astype("int8")
    counts = (
        temp.groupby(KEY_COLUMNS, observed=True)["_indicator"]
        .sum()
        .rename(f"_{loc_col}_diferenciada_sum")
    )
    parts.append(counts.to_frame())

    # Zero aqui é apenas acumulador parcial. A semântica de ausência é
    # restaurada em _finalize_aggregation por meio das contagens *_valid.
    for col in BINARY_COLUMNS:
        parts.append(grouped[col].sum(min_count=1).fillna(0).rename(f"_{col}_sum").to_frame())
        parts.append(grouped[col].count().rename(f"_{col}_valid").to_frame())

    for col in QUANTITATIVE_COLUMNS:
        parts.append(grouped[col].sum(min_count=1).fillna(0).rename(f"_{col}_sum").to_frame())
        parts.append(grouped[col].count().rename(f"_{col}_valid").to_frame())

    return pd.concat(parts, axis=1).fillna(0)


def _finalize_aggregation(accum: pd.DataFrame, year: int) -> pd.DataFrame:
    """Converte os acumuladores em features finais por município e rede."""
    out = pd.DataFrame(index=accum.index)
    qtd = accum["_qtd_escolas"].replace(0, np.nan)

    out["qtd_escolas"] = accum["_qtd_escolas"].astype("int64")
    out["pct_info_detalhada_completa"] = accum["_info_completa_sum"] / qtd * 100

    for col, categories in CATEGORICAL_COLUMNS.items():
        valid_count = accum[f"_{col}_valid"]
        valid_denominator = valid_count.replace(0, np.nan)
        name = col.lower()
        out[f"cobertura_{name}_pct"] = valid_count / qtd * 100
        for category in categories:
            out[f"pct_{name}_{category}"] = accum[f"_{col}_{category}_sum"] / valid_denominator * 100

    loc_col = LOCALIZACAO_DIFERENCIADA_COLUMN
    valid_count = accum[f"_{loc_col}_valid"]
    valid_denominator = valid_count.replace(0, np.nan)
    out["pct_localizacao_diferenciada"] = (
        accum[f"_{loc_col}_diferenciada_sum"] / valid_denominator * 100
    )
    out["cobertura_localizacao_diferenciada_pct"] = valid_count / qtd * 100

    for col in BINARY_COLUMNS:
        valid_count = accum[f"_{col}_valid"]
        valid_denominator = valid_count.replace(0, np.nan)
        name = col.lower()
        out[f"pct_{name}"] = accum[f"_{col}_sum"] / valid_denominator * 100
        out[f"cobertura_{name}_pct"] = valid_count / qtd * 100

    for col in QUANTITATIVE_COLUMNS:
        valid_count = accum[f"_{col}_valid"]
        valid_denominator = valid_count.replace(0, np.nan)
        name = col.lower()
        out[f"media_{name}"] = accum[f"_{col}_sum"] / valid_denominator
        out[f"total_{name}"] = accum[f"_{col}_sum"].where(valid_count > 0, np.nan)
        out[f"cobertura_{name}_pct"] = valid_count / qtd * 100

    matriculas = out["total_qt_mat_bas"].replace(0, np.nan)
    docentes = out["total_qt_doc_bas"].replace(0, np.nan)
    salas = out["total_qt_salas_utilizadas"].replace(0, np.nan)
    dispositivos = (
        out["total_qt_desktop_aluno"]
        + out["total_qt_comp_portatil_aluno"]
        + out["total_qt_tablet_aluno"]
    )

    out["alunos_por_docente"] = matriculas / docentes
    out["alunos_por_sala"] = matriculas / salas
    out["docentes_por_100_matriculas"] = docentes / matriculas * 100
    out["dispositivos_aluno_por_100_matriculas"] = dispositivos / matriculas * 100

    out = out.reset_index()
    out.insert(0, "ano_censo", year)
    out["rede"] = out["TP_DEPENDENCIA"].map(DEPENDENCIA_MAP)

    key_order = ["ano_censo", "CO_MUNICIPIO", "TP_DEPENDENCIA", "rede", "qtd_escolas"]
    out = out[key_order + [c for c in out.columns if c not in key_order]]

    numeric_cols = out.select_dtypes(include="number").columns.difference(["ano_censo", "qtd_escolas"])
    out[numeric_cols] = out[numeric_cols].round(6)

    return out.sort_values(["CO_MUNICIPIO", "TP_DEPENDENCIA"]).reset_index(drop=True)


def process_year(year: int, input_dir: Path, output_dir: Path, chunk_size: int) -> dict:
    """Processa uma edição anual e mantém apenas escolas em atividade."""
    zip_path = input_dir / f"microdados_censo_escolar_{year}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado para {year}: {zip_path}")

    csv_name = _validate_header(zip_path, year)

    print(f"\n[{year}] Fonte: {zip_path}")
    print(f"[{year}] CSV principal: {csv_name}")
    print(f"[{year}] Leitura em chunks de {chunk_size:,} linhas...")

    accumulator = None
    rows_read = 0
    rows_active = 0

    with ZipFile(zip_path) as zf:
        with zf.open(csv_name) as raw:
            reader = pd.read_csv(
                raw,
                sep=";",
                encoding="latin1",
                usecols=SELECTED_COLUMNS,
                dtype="string",
                keep_default_na=False,
                na_filter=False,
                chunksize=chunk_size,
                low_memory=False,
            )

            for i, chunk in enumerate(reader, start=1):
                rows_read += len(chunk)
                chunk = _normalize_strings(chunk)
                _validate_domains(chunk, year)

                # Dicionário oficial:
                # 1 = Em Atividade; 2 = Paralisada; 3 = Extinta.
                chunk = chunk.loc[
                    chunk[SITUACAO_FUNCIONAMENTO_COLUMN] == SITUACAO_EM_ATIVIDADE
                ].copy()

                rows_active += len(chunk)

                if chunk.empty:
                    continue

                accumulator = _add_frames(accumulator, _aggregate_chunk(chunk))

                if i == 1 or i % 5 == 0:
                    print(
                        f"[{year}] {rows_read:,} registros lidos | "
                        f"{rows_active:,} escolas ativas processadas..."
                    )

    if accumulator is None:
        raise RuntimeError(f"{year}: nenhuma escola em atividade foi processada.")

    final = _finalize_aggregation(accumulator, year)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"censo_escolar_municipio_rede_{year}.parquet"
    final.to_parquet(output_path, index=False)

    print(f"[{year}] Linhas de origem: {rows_read:,}")
    print(f"[{year}] Escolas em atividade: {rows_active:,}")
    print(f"[{year}] Registros excluídos por situação: {rows_read - rows_active:,}")
    print(f"[{year}] Grupos município/rede: {len(final):,}")
    print(f"[{year}] Colunas de saída: {len(final.columns)}")
    print(f"[{year}] Salvo em: {output_path}")

    return {
        "year": year,
        "input_zip": str(zip_path),
        "input_csv_inside_zip": csv_name,
        "input_rows": rows_read,
        "active_school_rows": rows_active,
        "excluded_non_active_rows": rows_read - rows_active,
        "output_file": str(output_path),
        "output_rows": len(final),
        "output_columns": len(final.columns),
    }


def write_manifest(output_dir: Path, results: list[dict], years: list[int]) -> Path:
    """Grava proveniência e regras metodológicas do processamento."""
    manifest = {
        "dataset": "Censo Escolar - características municipais por rede",
        "official_source": SOURCE_URL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_strategy": {
            "avaliacao_2023": "censo_escolar_2022",
            "avaliacao_2024": "censo_escolar_2023",
        },
        "school_filter": {
            "column": SITUACAO_FUNCIONAMENTO_COLUMN,
            "included_value": "1",
            "included_label": "Em Atividade",
            "excluded_values": {"2": "Paralisada", "3": "Extinta"},
            "rationale": (
                "Only active schools are used to characterize the educational "
                "context available in the census year."
            ),
        },
        "join_strategy": {
            "gold_key": ["id_municipio", "rede"],
            "censo_key": ["CO_MUNICIPIO", "rede"],
            "aggregation": ["CO_MUNICIPIO", "TP_DEPENDENCIA"],
            "dependencia_mapping": DEPENDENCIA_MAP,
        },
        "missing_data_policy": {
            "empty_values": "converted_to_null",
            "binary_proportions": "calculated only over active schools with valid information",
            "quantitative_means": "calculated only over active schools with valid information",
            "quantitative_totals_without_valid_values": "kept_as_null",
            "coverage_metrics": (
                "percentage of active schools with valid information; "
                "zero when no active school in the group has a valid value"
            ),
            "zero_imputation": False,
        },
        "selected_columns": SELECTED_COLUMNS,
        "temporal_harmonization": {
            "TP_LOCALIZACAO_DIFERENCIADA": (
                "harmonized as binary: 0 = not differentiated; "
                "1/2/3/8 = differentiated. Category 8 appears in 2023 and is "
                "grouped with the other differentiated categories to preserve "
                "a consistent meaning across years"
            )
        },
        "excluded_after_schema_validation": {
            "QT_FUNCIONARIOS": "absent from the 2023 main CSV; excluded to keep a consistent schema"
        },
        "years_processed": years,
        "outputs": results,
    }

    path = output_dir / "censo_escolar_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    """Executa o processamento das edições solicitadas."""
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    years = sorted(set(args.years))

    print("=== PREPARAÇÃO DO CENSO ESCOLAR ===")
    print(f"Entrada: {input_dir}")
    print(f"Saída:   {output_dir}")
    print(f"Anos:    {years}")

    results = [
        process_year(year, input_dir, output_dir, args.chunk_size)
        for year in years
    ]

    manifest = write_manifest(output_dir, results, years)

    print("\n=== CONCLUÍDO ===")
    print(f"Manifest: {manifest}")
    print("Regra temporal: avaliação 2023 <- Censo 2022 | avaliação 2024 <- Censo 2023")


if __name__ == "__main__":
    main()
