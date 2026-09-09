"""Constrói a base final de modelagem enriquecida com o Censo Escolar.

O módulo parte da camada Gold da Fase 2, restringe a população aos alunos
efetivamente avaliados e incorpora as características municipais do Censo
Escolar previamente processadas.

A estratégia temporal do projeto é preservada:
- avaliação de 2023 utiliza o Censo Escolar de 2022;
- avaliação de 2024 utiliza o Censo Escolar de 2023.

O módulo não realiza imputação, seleção de features nem transformação estatística
destinada ao modelo. Sua responsabilidade é materializar e validar a base
enriquecida que alimentará as etapas posteriores de machine learning.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================
# Caminhos resolvidos a partir da raiz do repositório.

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = DATA_DIR / "alunos_modelagem.parquet"

CENSO_FILES = {
    2022: EXTERNAL_DIR / "censo_escolar_municipio_rede_2022.parquet",
    2023: EXTERNAL_DIR / "censo_escolar_municipio_rede_2023.parquet",
}

OUTPUT_FILE = PROCESSED_DIR / "alunos_modelagem_enriquecido.parquet"


# ============================================================
# Expected invariants
# ============================================================
# Invariantes derivados da auditoria da Gold e da validação do enriquecimento.

EXPECTED_ROWS = {
    2023: 1_502_809,
    2024: 1_851_852,
}

EXPECTED_TOTAL_ROWS = sum(EXPECTED_ROWS.values())

EXPECTED_CENSO_FEATURES = 71


# ============================================================
# Population
# ============================================================

def load_modeling_population() -> pd.DataFrame:
    """Carrega a Gold e seleciona a população efetivamente avaliada.

    Returns:
        pd.DataFrame: População de modelagem validada para 2023 e 2024.

    Raises:
        RuntimeError: Se faltarem colunas obrigatórias, se as quantidades
        esperadas divergirem ou se surgirem anos não previstos.
    """
    print("Carregando camada Gold...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Linhas Gold: {len(df):,}")
    print(f"Colunas Gold: {len(df.columns)}")

    # Colunas mínimas necessárias para filtrar e enriquecer a população.
    required = {
        "ano",
        "id_municipio",
        "rede",
        "presenca",
        "preenchimento_caderno",
        "proficiencia",
        "alfabetizado",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Colunas obrigatórias ausentes na Gold: {sorted(missing)}"
        )

    # Mantém apenas alunos presentes, com prova preenchida e proficiência observada.
    modeling = df.loc[
        (df["presenca"] == "Presente")
        & (df["preenchimento_caderno"] == "Prova preenchida")
        & (df["proficiencia"].notna())
    ].copy()

    print(f"População avaliada: {len(modeling):,}")

    if len(modeling) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            "Quantidade inesperada na população de modelagem: "
            f"{len(modeling):,}. "
            f"Esperado: {EXPECTED_TOTAL_ROWS:,}."
        )

    counts = modeling["ano"].value_counts().to_dict()

    for year, expected in EXPECTED_ROWS.items():
        observed = counts.get(year, 0)

        if observed != expected:
            raise RuntimeError(
                f"{year}: {observed:,} registros encontrados; "
                f"esperado {expected:,}."
            )

    unexpected_years = set(counts) - set(EXPECTED_ROWS)

    if unexpected_years:
        raise RuntimeError(
            f"Anos inesperados na população: {sorted(unexpected_years)}"
        )

    return modeling


# ============================================================
# Census lookup
# ============================================================

def load_census_lookup() -> pd.DataFrame:
    """Carrega e valida as tabelas agregadas do Censo Escolar.

    Returns:
        pd.DataFrame: Lookup temporal com chave ano + id_municipio + rede.

    Raises:
        RuntimeError: Se houver inconsistências estruturais, duplicidades ou
        quantidade inesperada de features.
    """
    frames = []

    for census_year, file_path in CENSO_FILES.items():
        print(f"Carregando Censo Escolar {census_year}...")

        census = pd.read_parquet(file_path)

        required = {
            "ano_censo",
            "CO_MUNICIPIO",
            "TP_DEPENDENCIA",
            "rede",
        }

        missing = required - set(census.columns)

        if missing:
            raise RuntimeError(
                f"Censo {census_year}: colunas obrigatórias "
                f"ausentes: {sorted(missing)}"
            )

        if not (census["ano_censo"] == census_year).all():
            raise RuntimeError(
                f"Censo {census_year}: ano_censo inconsistente."
            )

        duplicates = census.duplicated(
            subset=["CO_MUNICIPIO", "rede"]
        ).sum()

        if duplicates:
            raise RuntimeError(
                f"Censo {census_year}: {duplicates} duplicidades "
                "em CO_MUNICIPIO + rede."
            )

        # As demais colunas devem corresponder às 71 features candidatas do Censo.
        technical_columns = {
            "ano_censo",
            "CO_MUNICIPIO",
            "TP_DEPENDENCIA",
            "rede",
        }

        feature_columns = [
            column
            for column in census.columns
            if column not in technical_columns
        ]

        if len(feature_columns) != EXPECTED_CENSO_FEATURES:
            raise RuntimeError(
                f"Censo {census_year}: "
                f"{len(feature_columns)} features encontradas; "
                f"esperado {EXPECTED_CENSO_FEATURES}."
            )

        # Regra temporal: avaliação 2023 <- Censo 2022; avaliação 2024 <- Censo 2023.
        census["ano"] = census_year + 1

        frames.append(census)

    lookup = pd.concat(frames, ignore_index=True)

    lookup = lookup.rename(
        columns={"CO_MUNICIPIO": "id_municipio"}
    )

    duplicates = lookup.duplicated(
        subset=["ano", "id_municipio", "rede"]
    ).sum()

    if duplicates:
        raise RuntimeError(
            f"Lookup temporal contém {duplicates} chaves duplicadas."
        )

    print(f"Lookup Censo: {len(lookup):,} grupos município/rede/ano")

    return lookup


# ============================================================
# Enrichment
# ============================================================

def enrich_population(
    modeling: pd.DataFrame,
    census_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Relaciona a população de modelagem às características do Censo.

    Args:
        modeling: População de alunos efetivamente avaliados.
        census_lookup: Dimensão temporal com as features agregadas do Censo.

    Returns:
        pd.DataFrame: População enriquecida com as características externas.

    Raises:
        RuntimeError: Se o merge alterar a quantidade de linhas, deixar alunos
        sem correspondência ou violar o mapeamento temporal esperado.
    """

    print("Relacionando população com o Censo Escolar...")

    # O merge many-to-one não pode alterar a quantidade de alunos.
    rows_before = len(modeling)

    enriched = modeling.merge(
        census_lookup,
        on=["ano", "id_municipio", "rede"],
        how="left",
        validate="many_to_one",
        indicator="_merge_censo",
    )

    if len(enriched) != rows_before:
        raise RuntimeError(
            "O merge alterou a quantidade de registros: "
            f"{rows_before:,} -> {len(enriched):,}."
        )

    # Valida cobertura integral do relacionamento com o Censo.
    coverage = enriched["_merge_censo"].eq("both")

    missing_rows = int((~coverage).sum())

    if missing_rows:
        raise RuntimeError(
            f"{missing_rows:,} alunos ficaram sem correspondência "
            "com o Censo Escolar."
        )

    print("Cobertura Censo: 100.00%")

    # Confirma explicitamente a defasagem temporal utilizada no enriquecimento.
    temporal_mapping = (
        enriched[
            ["ano", "ano_censo"]
        ]
        .drop_duplicates()
        .sort_values("ano")
    )

    expected_mapping = {
        2023: 2022,
        2024: 2023,
    }

    observed_mapping = dict(
        zip(
            temporal_mapping["ano"],
            temporal_mapping["ano_censo"],
        )
    )

    if observed_mapping != expected_mapping:
        raise RuntimeError(
            "Mapeamento temporal inesperado: "
            f"{observed_mapping}"
        )

    enriched = enriched.drop(
        columns="_merge_censo"
    )

    return enriched


# ============================================================
# Final validation
# ============================================================

def validate_output(df: pd.DataFrame) -> None:
    """Valida os principais invariantes do dataset enriquecido.

    Args:
        df: Dataset final após o relacionamento com o Censo Escolar.

    Raises:
        RuntimeError: Se quantidades esperadas divergirem ou houver registros
        sem ano_censo.
    """
    print("Validando dataset enriquecido...")

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Dataset final possui {len(df):,} linhas; "
            f"esperado {EXPECTED_TOTAL_ROWS:,}."
        )

    counts = df["ano"].value_counts().to_dict()

    for year, expected in EXPECTED_ROWS.items():
        if counts.get(year, 0) != expected:
            raise RuntimeError(
                f"Dataset final: quantidade inesperada para {year}."
            )

    if df["ano_censo"].isna().any():
        raise RuntimeError(
            "Existem registros sem ano_censo."
        )

    print("Validações finais concluídas.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Executa a construção e persistência da base enriquecida."""
    print("=== CONSTRUÇÃO DA BASE DE MODELAGEM ===")

    # Cria o diretório de saída apenas se necessário.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling = load_modeling_population()

    census_lookup = load_census_lookup()

    enriched = enrich_population(
        modeling,
        census_lookup,
    )

    validate_output(enriched)

    print("Salvando dataset enriquecido...")

    # Persiste a base somente após todas as validações.
    enriched.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== CONCLUÍDO ===")
    print(f"Linhas: {len(enriched):,}")
    print(f"Colunas: {len(enriched.columns)}")
    print(f"Arquivo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
