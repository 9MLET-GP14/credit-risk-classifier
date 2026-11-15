"""Testes iniciais de sanidade do ambiente e empacotamento."""

import credit_risk


def test_package_version() -> None:
    """Verifica se o pacote é importável e expõe a versão esperada."""
    assert credit_risk.__version__ == "0.1.0"
