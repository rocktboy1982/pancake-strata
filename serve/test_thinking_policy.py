"""STRATA_THINKING_POLICY (opt-in; Strata PR #5512 by rwkeyes, LEADS.md "Strata thinking skip for short answers").

The contract (the PR's): nothing at all when off; a client's own choice is never overridden; only a prompt that asks
for a short DIRECT answer loses its thinking; a code prompt never does; a classifier failure leaves the request alone.
"""
import unittest
from unittest import mock

from serve import prompt_kind as PK

DIRECT = "In the memo below, reply with the invoice number only:\n\nMemo: the invoice is INV-4471, dated Tuesday."
REASON = "Explain why the sky is blue, in one paragraph."
CODE = "Why does this panic?\n\nTraceback (most recent call last):\n  File \"main.py\", line 12\nKeyError: 'retries'"


def run(text, kwargs=None, req=None):
    kw = dict(kwargs or {})
    PK.apply_thinking_policy([{"role": "user", "content": text}], kw, req)
    return kw


class Policy(unittest.TestCase):
    def setUp(self):
        self.stats = mock.patch.dict(PK.STATS, {"direct": 0, "reason": 0, "applied": 0, "shadowed": 0,
                                                "last": None})
        self.stats.start()

    def tearDown(self):
        self.stats.stop()

    def mode(self, m):
        return mock.patch.object(PK, "_MODE", m)

    def test_default_env_is_off_and_changes_nothing(self):
        with mock.patch.object(PK, "_MODE", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(PK.policy_mode(), "off")
            self.assertEqual(run(DIRECT, {"reasoning_effort": "high"}), {"reasoning_effort": "high"})
            self.assertEqual(run(DIRECT), {})
        self.assertEqual(PK.STATS["direct"], 0)

    def test_typo_is_off(self):
        with mock.patch.object(PK, "_MODE", None), mock.patch.dict("os.environ", {"STRATA_THINKING_POLICY": "yes"}):
            self.assertEqual(PK.policy_mode(), "off")

    def test_on_turns_thinking_off_for_a_direct_prompt(self):
        with self.mode("on"):
            self.assertEqual(run(DIRECT, {"reasoning_effort": "high"}, req={}), {"enable_thinking": False})
            self.assertEqual(PK.STATS["applied"], 1)

    def test_reason_and_code_keep_thinking(self):
        with self.mode("on"):
            self.assertEqual(run(REASON), {})
            self.assertEqual(run(CODE), {})
        self.assertEqual(PK.STATS["applied"], 0)

    def test_client_choice_is_never_overridden(self):
        with self.mode("on"):
            self.assertEqual(run(DIRECT, {"reasoning_effort": "high"}, req={"reasoning_effort": "high"}),
                             {"reasoning_effort": "high"})
            self.assertEqual(run(DIRECT, {}, req={"chat_template_kwargs": {"enable_thinking": True}}), {})
            self.assertEqual(run(DIRECT, {}, req={"thinking": {"type": "enabled"}}), {})
            # the server filled the effort: the policy still applies
            self.assertEqual(run(DIRECT, {"reasoning_effort": "high"},
                                 req={"reasoning_effort": "high", "_effort_from_shared": True}),
                             {"enable_thinking": False})

    def test_shadow_only_counts(self):
        with self.mode("shadow"):
            self.assertEqual(run(DIRECT, req={}), {})
        self.assertEqual(PK.STATS["shadowed"], 1)

    def test_classifier_failure_leaves_the_request(self):
        with self.mode("on"), mock.patch.object(PK, "wants_direct_answer", side_effect=RuntimeError("x")):
            self.assertEqual(run(DIRECT, req={}), {})


if __name__ == "__main__":
    unittest.main()
