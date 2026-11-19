import json
import unittest

from serve import grammar
from serve.server import StrataEngine


class GrammarKey(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]

    def test_spec_follows_the_engine_option_or_strict(self):
        fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
        self.assertIsNone(grammar.grammar_spec(False, None, None, None, fmt))
        self.assertEqual(grammar.grammar_spec(True, None, None, None, fmt), {"kind": "json_schema", "schema": {"type": "object"}})
        strict = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}}}
        self.assertEqual(grammar.grammar_spec(False, None, None, None, strict)["kind"], "json_schema")
        self.assertEqual(grammar.grammar_spec(True, self.TOOLS, None, self.TOOLS, None)["tool_choice"], "auto")
        self.assertIsNone(grammar.grammar_spec(True, None, None, None, None))

    def test_engine_key_is_one_ascii_token(self):
        key = grammar.engine_key({"kind": "json_schema", "schema": {"enum": ["a b", "é"]}})
        self.assertTrue(key.startswith(" grammar="))
        self.assertNotIn(" ", key[1:])
        self.assertTrue(key.isascii())
        self.assertEqual(json.loads(key[len(" grammar="):])["schema"]["enum"], ["a b", "é"])
        self.assertEqual(grammar.engine_key(None), "")

    def test_sampling_keys_carry_the_grammar(self):
        keys = StrataEngine.sampling_keys({"strata_grammar": {"kind": "json_object"}})
        self.assertIn(' grammar={"kind":"json_object"}', keys)
        self.assertNotIn("grammar=", StrataEngine.sampling_keys({}))


if __name__ == "__main__":
    unittest.main()
