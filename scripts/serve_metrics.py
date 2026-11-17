"""Servidor HTTP leve para expor métricas Prometheus (/metrics) a partir de metrics.prom."""

import argparse
from http.server import HTTPServer
from pathlib import Path

from prometheus_client import MetricsHandler

from credit_risk.config import Settings


def serve_metrics(port: int = 8000, metrics_file: Path | None = None) -> None:
    """Inicia um servidor HTTP servindo o conteúdo do arquivo .prom na rota /metrics."""
    settings = Settings()
    path = metrics_file or settings.prometheus_metrics_path

    class CustomHandler(MetricsHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                if not path.exists():
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(
                        b"# Metricas ainda nao geradas. Execute um pipeline primeiro.\n"
                    )
                    return

                content = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Credit Risk Observability - Prometheus Exporter</h1>"
                    b"<p><a href='/metrics'>Visualizar metricas em /metrics</a></p></body></html>"
                )

    server = HTTPServer(("0.0.0.0", port), CustomHandler)
    print(f"Servidor de metricas Prometheus rodando em http://localhost:{port}/metrics")
    print(f"Lendo metricas de: {path.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor de metricas finalizado.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportador HTTP de métricas do Prometheus.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Porta HTTP do servidor (padrão: 8000)."
    )
    args = parser.parse_args()
    serve_metrics(port=args.port)
