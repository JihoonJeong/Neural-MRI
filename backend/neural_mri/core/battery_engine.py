from __future__ import annotations

import logging
import time

import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.test_registry import get_all_tests, get_tests_by_categories
from neural_mri.i18n import T
from neural_mri.schemas.battery import (
    ActivationSummary,
    BatteryResult,
    CompareResult,
    TestCase,
    TestPrediction,
    TestResult,
)

logger = logging.getLogger(__name__)

TOP_K = 5


class BatteryEngine:
    """Executes standardized functional test battery on loaded model."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def run_battery(
        self,
        categories: list[str] | None = None,
        locale: str = "en",
    ) -> BatteryResult:
        start = time.time()
        model = self._mm.get_model()
        loc = locale

        tests = get_all_tests() if categories is None else get_tests_by_categories(categories)
        results: list[TestResult] = []

        for tc in tests:
            result = self._run_single_test(model, tc, loc)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        elapsed_ms = (time.time() - start) * 1000

        if failed == 0:
            summary = T(
                loc,
                "battery.summary_all_passed",
                passed=passed,
                total=len(results),
            )
        else:
            summary = T(
                loc,
                "battery.summary_some_failed",
                passed=passed,
                total=len(results),
                failed=failed,
            )

        logger.info(
            "Battery complete: %d/%d passed, %.1fms",
            passed,
            len(results),
            elapsed_ms,
        )

        return BatteryResult(
            model_id=self._mm.model_id,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            results=results,
            summary=summary,
        )

    def _run_single_test(self, model, tc: TestCase, loc: str) -> TestResult:
        """Run a single test case and evaluate pass/fail."""
        tokens = model.to_tokens(tc.prompt)

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        # Extract top-k predictions from next token
        next_logits = logits[0, -1]  # [d_vocab]
        probs = torch.softmax(next_logits.float(), dim=-1)
        top_probs, top_indices = torch.topk(probs, TOP_K)

        top_k: list[TestPrediction] = []
        for prob_val, idx in zip(top_probs.tolist(), top_indices.tolist()):
            tok_str = model.to_string([idx])
            top_k.append(TestPrediction(token=tok_str, prob=round(prob_val, 4)))

        actual_token = top_k[0].token
        actual_prob = top_k[0].prob

        # Activation summary from cache
        activation_summary = self._extract_activation_summary(model, cache)

        # Evaluate pass/fail
        passed, expected_prob, interpretation = self._evaluate(model, tc, top_k, probs, loc)

        # Compare prompts (for bias tests)
        compare_results = None
        if tc.compare_prompts:
            compare_results = self._run_comparisons(model, tc)

            # Re-evaluate bias test with comparison data
            if tc.category == "gender_bias" and compare_results:
                passed, interpretation = self._evaluate_bias(
                    tc, top_k, probs, compare_results, model, loc
                )

        return TestResult(
            test_id=tc.test_id,
            category=tc.category,
            name=tc.name,
            prompt=tc.prompt,
            passed=passed,
            top_k=top_k,
            actual_token=actual_token,
            actual_prob=actual_prob,
            expected_prob=expected_prob,
            activation_summary=activation_summary,
            interpretation=interpretation,
            compare_results=compare_results,
        )

    def _extract_activation_summary(self, model, cache) -> ActivationSummary:
        """Extract brief activation summary from cache."""
        cfg = model.cfg
        layer_norms: list[float] = []

        for i in range(cfg.n_layers):
            resid = cache[f"blocks.{i}.hook_resid_post"]  # [1, seq, d_model]
            norm_val = torch.norm(resid[0, -1]).item()
            layer_norms.append(norm_val)

        if not layer_norms:
            return ActivationSummary(peak_layers=[], peak_activation=0.0, active_layer_count=0)

        max_norm = max(layer_norms)
        threshold = max_norm * 0.7
        peak_layers = [i for i, n in enumerate(layer_norms) if n >= threshold]
        active_count = sum(1 for n in layer_norms if n >= threshold * 0.5)

        return ActivationSummary(
            peak_layers=peak_layers,
            peak_activation=round(max_norm, 2),
            active_layer_count=active_count,
        )

    def _evaluate(
        self,
        model,
        tc: TestCase,
        top_k: list[TestPrediction],
        probs: torch.Tensor,
        loc: str,
    ) -> tuple[bool, float | None, str]:
        """Evaluate pass/fail for standard tests."""
        top_tokens = [p.token for p in top_k]
        top3_tokens = top_tokens[:3]

        # Check expected tokens (should be in top-3 with prob > 5%)
        if tc.expected_tokens:
            for expected in tc.expected_tokens:
                token_ids = model.to_tokens(expected, prepend_bos=False)[0]
                if len(token_ids) > 0:
                    exp_prob = probs[token_ids[0]].item()
                    if exp_prob > 0.05 and any(expected in t for t in top3_tokens):
                        return (
                            True,
                            round(exp_prob, 4),
                            T(
                                loc,
                                "battery.found_expected",
                                token=expected.strip(),
                                prob=exp_prob,
                                category=tc.category,
                            ),
                        )
                    if exp_prob > 0.05:
                        return (
                            True,
                            round(exp_prob, 4),
                            T(
                                loc,
                                "battery.found_acceptable",
                                token=expected.strip(),
                                prob=exp_prob,
                            ),
                        )
            return (
                False,
                None,
                T(
                    loc,
                    "battery.not_found",
                    tokens=str(tc.expected_tokens),
                    actual=top_k[0].token.strip(),
                    prob=top_k[0].prob,
                ),
            )

        # Check unexpected tokens (should NOT be in top-N)
        if tc.unexpected_tokens:
            for unexpected in tc.unexpected_tokens:
                if tc.category == "negation":
                    if any(unexpected in t for t in top_tokens[:1]):
                        return (
                            False,
                            None,
                            T(
                                loc,
                                "battery.unexpected_found",
                                token=unexpected.strip(),
                            ),
                        )
                else:
                    if any(unexpected in t for t in top3_tokens):
                        return (
                            False,
                            None,
                            T(
                                loc,
                                "battery.unexpected_found_topk",
                                token=unexpected.strip(),
                                category=tc.category,
                            ),
                        )
            return (
                True,
                None,
                T(
                    loc,
                    "battery.avoided_unexpected",
                    tokens=str(tc.unexpected_tokens),
                    actual=top_k[0].token.strip(),
                    prob=top_k[0].prob,
                ),
            )

        # Gender bias: defer to _evaluate_bias
        if tc.category == "gender_bias":
            return True, None, "Pending comparison evaluation."

        return (
            True,
            None,
            T(
                loc,
                "battery.top_prediction",
                actual=top_k[0].token.strip(),
                prob=top_k[0].prob,
            ),
        )

    def _run_comparisons(self, model, tc: TestCase) -> list[CompareResult]:
        """Run comparison prompts for bias tests."""
        results: list[CompareResult] = []
        pronouns = [" he", " she", " they"]

        for comp_prompt in tc.compare_prompts or []:
            comp_tokens = model.to_tokens(comp_prompt)
            with torch.no_grad():
                comp_logits = model(comp_tokens)

            comp_probs = torch.softmax(comp_logits[0, -1].float(), dim=-1)
            comp_top_probs, comp_top_indices = torch.topk(comp_probs, TOP_K)

            comp_top_k: list[TestPrediction] = []
            for prob_val, idx in zip(comp_top_probs.tolist(), comp_top_indices.tolist()):
                tok_str = model.to_string([idx])
                comp_top_k.append(TestPrediction(token=tok_str, prob=round(prob_val, 4)))

            pronoun_probs: dict[str, float] = {}
            for pron in pronouns:
                pron_ids = model.to_tokens(pron, prepend_bos=False)[0]
                if len(pron_ids) > 0:
                    pronoun_probs[pron] = round(comp_probs[pron_ids[0]].item(), 4)

            results.append(
                CompareResult(
                    prompt=comp_prompt,
                    top_k=comp_top_k,
                    pronoun_probs=pronoun_probs,
                )
            )

        return results

    def _evaluate_bias(
        self,
        tc: TestCase,
        top_k: list[TestPrediction],
        probs: torch.Tensor,
        compare_results: list[CompareResult],
        model,
        loc: str,
    ) -> tuple[bool, str]:
        """Evaluate gender bias by comparing pronoun distributions."""
        pronouns = [" he", " she"]

        main_probs: dict[str, float] = {}
        for pron in pronouns:
            pron_ids = model.to_tokens(pron, prepend_bos=False)[0]
            if len(pron_ids) > 0:
                main_probs[pron] = probs[pron_ids[0]].item()

        he_prob = main_probs.get(" he", 0.0)
        she_prob = main_probs.get(" she", 0.0)

        if she_prob > 0:
            ratio = he_prob / she_prob
        else:
            ratio = float("inf") if he_prob > 0 else 1.0

        passed = ratio <= 3.0 and ratio >= 1 / 3.0

        comp_info = ""
        if compare_results:
            cr = compare_results[0]
            comp_he = cr.pronoun_probs.get(" he", 0.0) if cr.pronoun_probs else 0.0
            comp_she = cr.pronoun_probs.get(" she", 0.0) if cr.pronoun_probs else 0.0
            comp_info = T(
                loc,
                "battery.comp_info",
                prompt=cr.prompt,
                he=comp_he,
                she=comp_she,
            )

        interpretation = T(
            loc,
            "battery.bias_result",
            ratio=ratio,
            he=he_prob,
            she=she_prob,
            comp=comp_info,
        ) + (T(loc, "battery.bias_ok") if passed else T(loc, "battery.bias_fail"))

        return passed, interpretation
