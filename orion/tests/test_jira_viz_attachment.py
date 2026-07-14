"""
Test proving that _attach_viz_to_jira cross-contaminates PR attachments.

When multiple PRs regress the same test, issue_keys_by_test_pull is keyed
only by test_name. The attachment loop iterates each pr_num but always uses
the same flat dict, attaching every PR's viz HTML to issues from *all* PRs.

CodeRabbit review: https://github.com/cloud-bulldozer/orion/pull/421
"""

import os
import tempfile
from unittest.mock import MagicMock, call

import pytest

from main import auto_create_jira_issues, _attach_viz_to_jira, build_viz_output_file


def _make_regression(test_name, uuid, metric, pct_change=-10.0):
    return {
        "test_name": test_name,
        "uuid": uuid,
        "metrics_with_change": [
            {"name": metric, "percentage_change": pct_change}
        ],
        "bad_ver": "4.18",
        "prev_ver": "4.17",
    }


class TestCrossPRAttachmentBug:
    """Proves that pull viz files leak across PRs sharing the same test name."""

    def test_attach_viz_uses_same_keys_for_all_prs(self):
        """Two PRs regress the same test. Each PR's viz gets attached to
        the other PR's JIRA issue because issue_keys_by_test_pull is
        keyed only by test_name, not by (pr_num, test_name).

        This test PASSES today, proving the bug exists — the mock provider
        receives attach_file calls that pair the wrong PR's HTML with
        the wrong JIRA issue.
        """
        provider = MagicMock()

        # Simulate: PR 1111 and PR 2222 both regress "node-density"
        # auto_create_jira_issues is called once with ALL pull regressions
        # flattened, producing a single dict keyed by test_name.
        pr1_regression = _make_regression("node-density", "uuid-pr1111", "podLatency")
        pr2_regression = _make_regression("node-density", "uuid-pr2222", "podLatency")

        # Provider.create_ack returns different JIRA keys per call
        provider.create_ack = MagicMock(side_effect=["PERF-100", "PERF-200"])

        # This is exactly what main.py does: flatten all PR regressions into one list
        all_pull_regressions = [pr1_regression, pr2_regression]
        _, issue_keys_by_test_pull = auto_create_jira_issues(
            all_pull_regressions, provider, MagicMock()
        )

        # issue_keys_by_test_pull is {"node-density": ["PERF-100", "PERF-200"]}
        # PERF-100 was created for PR 1111, PERF-200 for PR 2222
        assert issue_keys_by_test_pull == {
            "node-density": ["PERF-100", "PERF-200"]
        }

        # Create fake viz files
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "output")
            pull_numbers = [1111, 2222]

            # Create the viz HTML files that would be generated
            for pr_num in pull_numbers:
                viz_path = build_viz_output_file(base, "node-density", f"pull_{pr_num}")
                with open(viz_path, "w") as f:
                    f.write(f"<html>viz for PR {pr_num}</html>")

            # This is the actual loop from main.py lines 765-767:
            # for pr_num in kwargs.get("pull_numbers", []):
            #     _attach_viz_to_jira(jira_provider, issue_keys_by_test_pull, ...)
            provider.attach_file = MagicMock(return_value=True)
            for pr_num in pull_numbers:
                _attach_viz_to_jira(
                    provider, issue_keys_by_test_pull, base,
                    f"pull_{pr_num}", MagicMock()
                )

            # Collect all attach_file calls
            attach_calls = provider.attach_file.call_args_list

            # BUG: PR 1111's viz is attached to PERF-200 (PR 2222's issue)
            # and PR 2222's viz is attached to PERF-100 (PR 1111's issue)
            pr1_viz = build_viz_output_file(base, "node-density", "pull_1111")
            pr2_viz = build_viz_output_file(base, "node-density", "pull_2222")

            # Each PR's viz should only attach to its own issue.
            # Correct behavior: 2 calls total (1 per PR × 1 issue each)
            # Actual behavior: 4 calls (each PR's viz → both issues)
            assert len(attach_calls) == 4, (
                f"Expected 4 cross-contaminated calls (the bug), got {len(attach_calls)}"
            )

            # Verify the cross-contamination: PR 1111's viz goes to PERF-200
            assert call("PERF-200", pr1_viz) in attach_calls, (
                "BUG: PR 1111 viz should NOT be attached to PERF-200 (PR 2222's issue), "
                "but current code does exactly this"
            )
            # And PR 2222's viz goes to PERF-100
            assert call("PERF-100", pr2_viz) in attach_calls, (
                "BUG: PR 2222 viz should NOT be attached to PERF-100 (PR 1111's issue), "
                "but current code does exactly this"
            )

    def test_correct_behavior_would_scope_by_pr(self):
        """If the bug were fixed, each PR's viz would only attach to its
        own JIRA issues. This test documents what correct behavior looks like
        and is expected to FAIL with the current code.
        """
        provider = MagicMock()

        pr1_regression = _make_regression("node-density", "uuid-pr1111", "podLatency")
        pr2_regression = _make_regression("node-density", "uuid-pr2222", "podLatency")

        provider.create_ack = MagicMock(side_effect=["PERF-100", "PERF-200"])

        all_pull_regressions = [pr1_regression, pr2_regression]
        _, issue_keys_by_test_pull = auto_create_jira_issues(
            all_pull_regressions, provider, MagicMock()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "output")
            pull_numbers = [1111, 2222]

            for pr_num in pull_numbers:
                viz_path = build_viz_output_file(base, "node-density", f"pull_{pr_num}")
                with open(viz_path, "w") as f:
                    f.write(f"<html>viz for PR {pr_num}</html>")

            provider.attach_file = MagicMock(return_value=True)
            for pr_num in pull_numbers:
                _attach_viz_to_jira(
                    provider, issue_keys_by_test_pull, base,
                    f"pull_{pr_num}", MagicMock()
                )

            attach_calls = provider.attach_file.call_args_list

            pr1_viz = build_viz_output_file(base, "node-density", "pull_1111")
            pr2_viz = build_viz_output_file(base, "node-density", "pull_2222")

            # Correct behavior: each PR's viz attached only to its own issue
            # PERF-100 (from PR 1111) should only get pull_1111 viz
            # PERF-200 (from PR 2222) should only get pull_2222 viz
            pr1_to_wrong_issue = call("PERF-200", pr1_viz) in attach_calls
            pr2_to_wrong_issue = call("PERF-100", pr2_viz) in attach_calls

            assert not pr1_to_wrong_issue, (
                "PR 1111 viz was attached to PERF-200 (PR 2222's issue) — cross-contamination!"
            )
            assert not pr2_to_wrong_issue, (
                "PR 2222 viz was attached to PERF-100 (PR 1111's issue) — cross-contamination!"
            )
