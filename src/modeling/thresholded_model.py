"""Wrapper de classificação binária com limiar de decisão explícito.

Este módulo fornece um artefato persistível que combina um estimador
probabilístico já ajustado com o threshold operacional selecionado durante o
desenvolvimento do modelo. O objetivo é garantir que ``predict`` reproduza a
regra de decisão utilizada na avaliação e na aplicação estratégica, enquanto
``predict_proba`` permanece delegado ao pipeline original.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ThresholdedBinaryClassifier:
    """Combina um estimador probabilístico ajustado com um threshold binário.

    A classe foi projetada para envolver o pipeline final já treinado. O
    pré-processamento e o estimador probabilístico permanecem encapsulados no
    objeto ``estimator``; apenas a regra de ``predict`` é substituída pelo
    limiar explicitamente selecionado em previsões out-of-fold.

    Args:
        estimator: Estimador binário já ajustado que implemente
            ``predict_proba``.
        threshold: Limiar aplicado à probabilidade da classe positiva. Deve
            pertencer ao intervalo fechado [0, 1].

    Raises:
        TypeError: Se o estimador não implementar ``predict_proba``.
        ValueError: Se o threshold estiver fora do intervalo [0, 1].
    """

    def __init__(self, estimator: Any, threshold: float) -> None:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("O estimador deve implementar predict_proba().")

        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("O threshold deve pertencer ao intervalo [0, 1].")

        self.estimator = estimator
        self.threshold = threshold

    @property
    def classes_(self) -> np.ndarray:
        """Retorna as classes aprendidas pelo estimador encapsulado."""
        return self.estimator.classes_

    def predict_proba(self, X: Any) -> np.ndarray:
        """Retorna as probabilidades produzidas pelo estimador encapsulado.

        Args:
            X: Matriz ou DataFrame com os preditores esperados pelo pipeline.

        Returns:
            np.ndarray: Probabilidades por classe, sem alteração em relação ao
            estimador original.
        """
        return self.estimator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Classifica observações usando o threshold operacional persistido.

        A classe positiva é assumida como a segunda coluna retornada por
        ``predict_proba`` (índice 1), em conformidade com o contrato binário do
        projeto: 0 = alfabetizado e 1 = não alfabetizado.

        Args:
            X: Matriz ou DataFrame com os preditores esperados pelo pipeline.

        Returns:
            np.ndarray: Vetor binário ``int8`` com a regra
            ``probabilidade >= threshold``.

        Raises:
            RuntimeError: Se ``predict_proba`` não retornar exatamente duas
            colunas de probabilidade.
        """
        probabilities = self.predict_proba(X)

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError(
                "O estimador encapsulado deve retornar probabilidades para "
                "exatamente duas classes."
            )

        return (probabilities[:, 1] >= self.threshold).astype("int8")
