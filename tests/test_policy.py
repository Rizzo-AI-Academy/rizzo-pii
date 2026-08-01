# -*- coding: utf-8 -*-
"""
Test della policy di anonimizzazione (src/app/policy.py).

Solo stdlib: si eseguono senza torch/transformers/flask e senza modello.

    python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))

import policy  # noqa: E402

TAXONOMY = {"FULLNAME", "AGE", "GENDER", "DATE", "TIME", "AMOUNT", "CF", "EMAIL", "IBAN"}


class PolicyTestCase(unittest.TestCase):
    """Isola i test dall'ambiente reale: niente env ereditate, config dir temporanea."""

    def setUp(self):
        self._env = {k: os.environ.pop(k) for k in ("PII_PROFILE", "PII_KEEP_TAGS")
                     if k in os.environ}
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_config_dir = policy.server_config.config_dir
        policy.server_config.config_dir = lambda: Path(self._tmp.name)
        self.warnings = []

    def tearDown(self):
        policy.server_config.config_dir = self._orig_config_dir
        self._tmp.cleanup()
        os.environ.update(self._env)
        for k in ("PII_PROFILE", "PII_KEEP_TAGS"):
            if k not in self._env:
                os.environ.pop(k, None)

    def load(self, **kwargs):
        kwargs.setdefault("warn", self.warnings.append)
        return policy.load_policy(**kwargs)


class TestParseTags(PolicyTestCase):

    def test_parse_tags_string_normalizes_case_and_spaces(self):
        self.assertEqual(policy.parse_tags(" age, gender "), ("AGE", "GENDER"))

    def test_parse_tags_accepts_list_and_semicolon(self):
        self.assertEqual(policy.parse_tags(["age", "AGE"]), ("AGE",))
        self.assertEqual(policy.parse_tags("age;gender"), ("AGE", "GENDER"))

    def test_parse_tags_empty_sources_give_empty_tuple(self):
        for raw in (None, "", "   ", [], ",,"):
            self.assertEqual(policy.parse_tags(raw), (), f"input: {raw!r}")


class TestPolicyDecision(PolicyTestCase):

    def test_default_policy_masks_every_tag(self):
        p = self.load()
        self.assertEqual(p.profile, policy.DEFAULT_PROFILE)
        self.assertEqual(p.keep_tags, frozenset())
        for tag in TAXONOMY:
            self.assertEqual(p.action(tag), policy.ACTION_MASK)

    def test_kept_tag_is_case_insensitive(self):
        p = policy.Policy(keep_tags=["age"])
        self.assertTrue(p.keeps("AGE"))
        self.assertTrue(p.keeps("age"))
        self.assertFalse(p.keeps("FULLNAME"))

    def test_as_dict_is_sorted_and_serializable(self):
        p = policy.Policy(keep_tags=["GENDER", "AGE"], profile="clinical")
        self.assertEqual(p.as_dict(), {"profile": "clinical", "keep_tags": ["AGE", "GENDER"]})


class TestProfiles(PolicyTestCase):

    def test_clinical_profile_keeps_its_tags_and_masks_the_rest(self):
        p = self.load(cli_profile="clinical", known_tags=TAXONOMY)
        for tag in ("AGE", "GENDER", "DATE", "TIME"):
            self.assertTrue(p.keeps(tag), tag)
        self.assertFalse(p.keeps("FULLNAME"))

    def test_explicit_tags_are_added_to_the_profile_tags(self):
        p = self.load(cli_profile="clinical", cli_keep_tags="AMOUNT", known_tags=TAXONOMY)
        self.assertTrue(p.keeps("AMOUNT"))
        self.assertTrue(p.keeps("AGE"))

    def test_unknown_profile_falls_back_to_default_with_a_warning(self):
        p = self.load(cli_profile="inesistente")
        self.assertEqual(p.keep_tags, frozenset())
        self.assertTrue(any("profilo sconosciuto" in w for w in self.warnings))


class TestPrecedence(PolicyTestCase):

    def test_cli_beats_env_and_file(self):
        os.environ["PII_KEEP_TAGS"] = "GENDER"
        policy.save_file("full", ["DATE"])
        p = self.load(cli_keep_tags="AGE", known_tags=TAXONOMY)
        self.assertEqual(p.keep_tags, frozenset({"AGE"}))

    def test_env_beats_file(self):
        os.environ["PII_KEEP_TAGS"] = "GENDER"
        policy.save_file("full", ["DATE"])
        p = self.load(known_tags=TAXONOMY)
        self.assertEqual(p.keep_tags, frozenset({"GENDER"}))

    def test_file_is_used_when_cli_and_env_are_absent(self):
        policy.save_file("full", ["DATE"])
        p = self.load(known_tags=TAXONOMY)
        self.assertEqual(p.keep_tags, frozenset({"DATE"}))

    def test_profile_precedence_is_independent_from_tags(self):
        os.environ["PII_PROFILE"] = "clinical"
        p = self.load(cli_keep_tags="AMOUNT", known_tags=TAXONOMY)
        self.assertEqual(p.profile, "clinical")
        self.assertTrue(p.keeps("AGE"))
        self.assertTrue(p.keeps("AMOUNT"))

    def test_save_and_load_file_round_trip(self):
        policy.save_file("clinical", " amount , age ")
        self.assertEqual(policy.load_file(),
                         {"profile": "clinical", "keep_tags": ["AMOUNT", "AGE"]})

    def test_corrupt_file_is_ignored(self):
        policy.policy_path().write_text("{non json", "utf-8")
        p = self.load(known_tags=TAXONOMY)
        self.assertEqual(p.keep_tags, frozenset())


class TestValidation(PolicyTestCase):

    def test_unknown_tag_is_dropped_with_a_warning(self):
        p = self.load(cli_keep_tags="AGE,NONESISTE", known_tags=TAXONOMY)
        self.assertEqual(p.keep_tags, frozenset({"AGE"}))
        self.assertTrue(any("NONESISTE" in w for w in self.warnings))

    def test_without_known_tags_no_validation_happens(self):
        p = self.load(cli_keep_tags="QUALSIASI")
        self.assertTrue(p.keeps("QUALSIASI"))

    def test_direct_identifier_left_in_clear_is_warned_about(self):
        self.load(cli_keep_tags="CF", known_tags=TAXONOMY)
        self.assertTrue(any("IN CHIARO" in w and "CF" in w for w in self.warnings))

    def test_no_warning_for_low_risk_tags(self):
        self.load(cli_profile="clinical", known_tags=TAXONOMY)
        self.assertEqual(self.warnings, [])


if __name__ == "__main__":
    unittest.main()
