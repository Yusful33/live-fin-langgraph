"""
Tests for example_question in run_ipo_dataset_experiment.

These guard against bare customer IDs (CUST<NNN>) being passed to the
LangGraph agent as if they were real queries.  Before the fix, any non-empty
string stored in the 'question' field — including bare IDs — was returned
unchanged, causing the Supervisor_Agent to route 100 % of calls to FINISH
with no agents invoked.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_ipo_dataset_experiment import example_question


# ---------------------------------------------------------------------------
# Bare customer-ID cases — must all return None so the row is skipped
# ---------------------------------------------------------------------------


def test_bare_cust_id_top_level_returns_none():
    assert example_question({"question": "CUST539"}) is None


def test_bare_cust_id_in_additional_properties_returns_none():
    assert example_question({"additional_properties": {"question": "CUST130"}}) is None


def test_bare_cust_id_lowercase_returns_none():
    assert example_question({"question": "cust042"}) is None


def test_bare_cust_id_padded_with_whitespace_returns_none():
    assert example_question({"question": "  CUST311  "}) is None


# ---------------------------------------------------------------------------
# Legitimate question cases — must be returned unchanged
# ---------------------------------------------------------------------------


def test_real_question_top_level_returned():
    q = "What is Apple's current P/E ratio?"
    assert example_question({"question": q}) == q


def test_real_question_in_additional_properties_returned():
    q = "How has Tesla's net income changed over the last five years?"
    row = {"additional_properties": {"question": q}}
    assert example_question(row) == q


def test_question_with_cust_in_middle_is_not_rejected():
    # A real sentence that happens to contain "CUST" should not be filtered.
    q = "How do customer (CUST) retention metrics affect MRR?"
    assert example_question({"question": q}) == q


# ---------------------------------------------------------------------------
# Missing / empty question cases — must return None so the row is skipped
# ---------------------------------------------------------------------------


def test_empty_row_returns_none():
    assert example_question({}) is None


def test_empty_question_string_returns_none():
    assert example_question({"question": ""}) is None


def test_none_question_returns_none():
    assert example_question({"question": None}) is None


def test_whitespace_only_question_returns_none():
    assert example_question({"question": "   "}) is None
