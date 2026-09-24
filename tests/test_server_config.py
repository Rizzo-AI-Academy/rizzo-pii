# -*- coding: utf-8 -*-
"""`server_config.resolve()`: catena CLI > env > config.json > default.

Due errori opposti da evitare: con `a or b` lo 0 esplicito (`--port 0`) veniva
scartato; con il solo `is not None` una variabile presente ma vuota (`PII_PORT=`)
finiva in `int("")` e il backend non partiva piu'.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import server_config  # noqa: E402


def risolvi(cfg=None, env=None, **cli):
    env = env or {}
    pulito = {k: v for k, v in os.environ.items() if k not in ("PII_HOST", "PII_PORT")}
    with patch.object(server_config, "load_config", return_value=cfg or {}), \
         patch.dict(os.environ, {**pulito, **env}, clear=True):
        return server_config.resolve(**cli)


class Resolve(unittest.TestCase):
    def test_default(self):
        self.assertEqual(risolvi(), (server_config.DEFAULT_HOST, server_config.DEFAULT_PORT))

    def test_precedenza(self):
        cfg = {"host": "10.0.0.1", "port": 6000}
        self.assertEqual(risolvi(cfg), ("10.0.0.1", 6000))
        self.assertEqual(risolvi(cfg, {"PII_HOST": "0.0.0.0", "PII_PORT": "7000"}),
                         ("0.0.0.0", 7000))
        self.assertEqual(risolvi(cfg, {"PII_PORT": "7000"}, cli_host="::1", cli_port=8000),
                         ("::1", 8000))

    def test_porta_zero_esplicita_rispettata(self):
        self.assertEqual(risolvi({"port": 6000}, cli_port=0)[1], 0)

    def test_variabile_vuota_ignorata(self):
        self.assertEqual(risolvi({"port": 6000}, {"PII_PORT": "", "PII_HOST": " "}),
                         (server_config.DEFAULT_HOST, 6000))


if __name__ == "__main__":
    unittest.main()
