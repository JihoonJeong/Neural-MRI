from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.test_registry import get_all_tests, get_tests_by_categories
from neural_mri.i18n import T
from neural_mri.schemas.battery import (
    ActivationSummary,
    BatteryResult,
    BatterySAESummary,
    CompareResult,
    CrossTestFeature,
    SAEFeatureBrief,
    TestCase,
    TestPrediction,
    TestResult,
    TestSAEResult,
)

if TYPE_CHECKING:
    from neural_mri.core.sae_manager import SAEManager

logger = logging.getLogger(__name__)

TOP_K = 5
SAE_BATTERY_TOP_K = 5


class BatteryEngine:
    """Executes standardized functional test battery on loaded model."""

    def __init__(
        self,
        model_manager: ModelManager,
        sae_manager: SAEManager | None = None,
    ) -> None:
        self._mm = model_manager
        self._sae_mgr = sae_manager

    def run_battery(
        self,
        categories: list[str] | None = None,
        locale: str = "en",
        include_sae: bool = False,
        sae_layer: int | None = None,
    ) -> BatteryResult:
        start = time.time()
        model = self._mm.get_model()
        loc = locale

        # Resolve SAE availability
        sae = None
        sae_info: dict | None = None
        resolved_layer: int | None = None
        hook_name: str | None = None

        if include_sae and self._sae_mgr is not None:
            from neural_mri.core.sae_registry import get_sae_info

            sae_info = get_sae_info(self._mm.model_id)
            if sae_info is not None:
                resolved_layer = sae_layer if sae_layer is not None else model.cfg.n_layers // 2
                if resolved_layer not in sae_info["layers"]:
                    resolved_layer = sae_info["layers"][len(sae_info["layers"]) // 2]
                device = str(model.cfg.device)
                sae = self._sae_mgr.get_sae(self._mm.model_id, resolved_layer, device)
                hook_name_from_meta = (
                    sae.cfg.metadata.get("hook_name") if sae.cfg.metadata else None
                )
                hook_name = hook_name_from_meta or sae_info["sae_id_template"].format(
                    layer=resolved_layer
                )

        tests = get_all_tests() if categories is None else get_tests_by_categories(categories)
        results: list[TestResult] = []
        all_test_sae_features: dict[str, list[SAEFeatureBrief]] = {}

        for tc in tests:
            result = self._run_single_test(
                model,
                tc,
                loc,
                sae=sae,
                hook_name=hook_name,
                sae_layer=resolved_layer,
                sae_info=sae_info,
            )
            results.append(result)
            if result.sae_features is not None:
                all_test_sae_features[result.test_id] = result.sae_features.top_features

        # Cross-test SAE summary
        sae_summary = None
        if (
            sae is not None
            and all_test_sae_features
            and sae_info is not None
            and resolved_layer is not None
        ):
            sae_summary = self._build_cross_test_summary(
                all_test_sae_features,
                results,
                resolved_layer,
                sae_info,
                loc,
            )

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
            "Battery complete: %d/%d passed, %.1fms%s",
            passed,
            len(results),
            elapsed_ms,
            f", SAE layer {resolved_layer}" if sae is not None else "",
        )

        return BatteryResult(
            model_id=self._mm.model_id,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            results=results,
            summary=summary,
            sae_summary=sae_summary,
        )

    def _run_single_test(
        self,
        model,
        tc: TestCase,
        loc: str,
        sae=None,
        hook_name: str | None = None,
        sae_layer: int | None = None,
        sae_info: dict | None = None,
    ) -> TestResult:
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

        # SAE feature extraction from the same cache
        sae_result = None
        if (
            sae is not None
            and hook_name is not None
            and sae_layer is not None
            and sae_info is not None
        ):
            sae_result = self._extract_sae_features(
                cache,
                sae,
                hook_name,
                sae_layer,
                sae_info,
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
            sae_features=sae_result,
        )

    def _extract_sae_features(
        self,
        cache,
        sae,
        hook_name: str,
        layer_idx: int,
        sae_info: dict,
    ) -> TestSAEResult:
        """Extract top SAE features at the last (prediction) token position."""
        activations = cache[hook_name]  # [1, seq_len, d_model]
        last_token_act = activations[0, -1:].float()  # [1, d_model]

        features = sae.encode(last_token_act)  # [1, d_sae]
        features_1d = features[0]  # [d_sae]

        topk_vals, topk_idxs = torch.topk(features_1d, SAE_BATTERY_TOP_K)
        global_max = features_1d.max().item()
        global_max = max(global_max, 1e-8)

        neuronpedia_template = sae_info.get("neuronpedia_url_template")

        top_features: list[SAEFeatureBrief] = []
        for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            np_url = None
            if neuronpedia_template:
                np_url = neuronpedia_template.format(layer=layer_idx, feature_idx=idx)
            top_features.append(
                SAEFeatureBrief(
                    feature_idx=idx,
                    activation=round(val, 4),
                    activation_normalized=round(val / global_max, 4),
                    neuronpedia_url=np_url,
                )
            )

        return TestSAEResult(
            layer_idx=layer_idx,
            hook_name=hook_name,
            top_features=top_features,
        )

    def _build_cross_test_summary(
        self,
        all_test_sae_features: dict[str, list[SAEFeatureBrief]],
        results: list[TestResult],
        layer_idx: int,
        sae_info: dict,
        loc: str,
    ) -> BatterySAESummary:
        """Analyze which SAE features appear across multiple tests."""
        test_category: dict[str, str] = {r.test_id: r.category for r in results}

        # feature_idx -> [(test_id, category, activation)]
        feature_map: dict[int, list[tuple[str, str, float]]] = {}
        all_unique: set[int] = set()

        for test_id, features in all_test_sae_features.items():
            cat = test_category.get(test_id, "unknown")
            for f in features:
                if f.activation > 0:
                    all_unique.add(f.feature_idx)
                    feature_map.setdefault(f.feature_idx, []).append((test_id, cat, f.activation))

        # Cross-test features (appearing in 2+ tests)
        neuronpedia_template = sae_info.get("neuronpedia_url_template")
        cross_features: list[CrossTestFeature] = []
        for feat_idx, appearances in sorted(feature_map.items(), key=lambda x: -len(x[1])):
            if len(appearances) < 2:
                continue
            test_ids = [a[0] for a in appearances]
            categories = sorted(set(a[1] for a in appearances))
            activations = [a[2] for a in appearances]
            np_url = None
            if neuronpedia_template:
                np_url = neuronpedia_template.format(layer=layer_idx, feature_idx=feat_idx)
            cross_features.append(
                CrossTestFeature(
                    feature_idx=feat_idx,
                    neuronpedia_url=np_url,
                    test_ids=test_ids,
                    categories=categories,
                    count=len(test_ids),
                    avg_activation=round(sum(activations) / len(activations), 4),
                    max_activation=round(max(activations), 4),
                )
            )

        # Per-category top features (top 3 per category)
        cat_features: dict[str, dict[int, float]] = {}
        for test_id, features in all_test_sae_features.items():
            cat = test_category.get(test_id, "unknown")
            cat_features.setdefault(cat, {})
            for f in features:
                if f.activation > 0:
                    cat_features[cat][f.feature_idx] = max(
                        cat_features[cat].get(f.feature_idx, 0),
                        f.activation,
                    )
        per_category_top: dict[str, list[int]] = {
            cat: [idx for idx, _ in sorted(feats.items(), key=lambda x: -x[1])[:3]]
            for cat, feats in cat_features.items()
        }

        # Build interpretation
        total_tests = len(all_test_sae_features)
        if cross_features:
            top_cross = cross_features[0]
            interpretation = T(
                loc,
                "battery.sae_cross_summary",
                n_cross=len(cross_features),
                top_idx=top_cross.feature_idx,
                top_count=top_cross.count,
                total=total_tests,
                n_unique=len(all_unique),
            )
        else:
            interpretation = T(
                loc,
                "battery.sae_no_cross",
                n_unique=len(all_unique),
                total=total_tests,
            )

        return BatterySAESummary(
            layer_idx=layer_idx,
            d_sae=sae_info["d_sae"],
            total_unique_features=len(all_unique),
            cross_test_features=cross_features,
            per_category_top_features=per_category_top,
            interpretation=interpretation,
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
