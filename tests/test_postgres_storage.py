import os
import unittest
from unittest.mock import patch

import postgres_storage


class PostgresStorageConfigTests(unittest.TestCase):
    def test_csv_mode_when_database_url_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(postgres_storage.postgres_enabled())
            with self.assertRaisesRegex(RuntimeError, "CSV mode remains active"):
                postgres_storage._connect()

    def test_postgres_enabled_when_database_url_is_set(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://example.invalid/db"}):
            self.assertTrue(postgres_storage.postgres_enabled())

    def test_whitespace_database_url_does_not_enable_postgres(self):
        with patch.dict(os.environ, {"DATABASE_URL": "   "}):
            self.assertFalse(postgres_storage.postgres_enabled())


if __name__ == "__main__":
    unittest.main()
