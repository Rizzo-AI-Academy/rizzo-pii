import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "app"))
from registry import Registry  # noqa: E402


class RegistryTest(unittest.TestCase):
    def test_same_value_same_placeholder(self):
        r = Registry()
        a = r.assign("EMAIL", "mario.rossi@example.com")
        b = r.assign("EMAIL", "  Mario.Rossi@Example.com ")
        self.assertEqual(a, "[EMAIL_1]")
        self.assertEqual(a, b)

    def test_first_tag_wins(self):
        r = Registry()
        a = r.assign("FULLNAME", "Acme Rossi")
        b = r.assign("ORG", "acme rossi")
        self.assertEqual(a, b)
        self.assertEqual(r.assign("ORG", "Altra Srl"), "[ORG_1]")

    def test_numbering_per_label(self):
        r = Registry()
        self.assertEqual(r.assign("IPADDR", "10.0.0.1"), "[IPADDR_1]")
        self.assertEqual(r.assign("IPADDR", "10.0.0.2"), "[IPADDR_2]")
        self.assertEqual(r.assign("EMAIL", "a@b.it"), "[EMAIL_1]")
        self.assertEqual(r.mapping, {"[IPADDR_1]": "10.0.0.1", "[IPADDR_2]": "10.0.0.2", "[EMAIL_1]": "a@b.it"})

    def test_no_mapping(self):
        r = Registry(mapping_enabled=False)
        self.assertEqual(r.assign("EMAIL", "a@b.it"), r.assign("EMAIL", "A@B.IT"))
        self.assertEqual(r.mapping, {})


if __name__ == "__main__":
    unittest.main()
