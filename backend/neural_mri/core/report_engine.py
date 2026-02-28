from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from neural_mri.core.analysis_engine import AnalysisEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.i18n import T
from neural_mri.schemas.report import (
    DiagnosticReport,
    ReportFinding,
    ReportImpression,
    ReportRequest,
)
from neural_mri.schemas.scan import (
    ActivationData,
    ActivationScanRequest,
    AnomalyData,
    AnomalyScanRequest,
    CircuitData,
    CircuitScanRequest,
    SAEData,
    StructuralData,
    WeightData,
)

logger = logging.getLogger(__name__)

ALL_MODES = ["T1", "T2", "fMRI", "DTI", "FLAIR"]


class ReportEngine:
    """Generates diagnostic reports by analyzing scan results (rule-based)."""

    def __init__(self, model_manager: ModelManager, analysis_engine: AnalysisEngine) -> None:
        self._mm = model_manager
        self._engine = analysis_engine

    def generate(self, req: ReportRequest) -> DiagnosticReport:
        start = time.time()
        model_info = self._mm.get_model_info()
        prompt = req.prompt or "The capital of France is"
        modes = req.include_modes or ALL_MODES
        loc = req.locale or "en"

        # Gather scan data — reuse cached or run fresh
        t1 = self._get_t1(req.cached_t1) if "T1" in modes else None
        t2 = self._get_t2(req.cached_t2) if "T2" in modes else None
        fmri = self._get_fmri(req.cached_fmri, prompt) if "fMRI" in modes else None
        dti = self._get_dti(req.cached_dti, prompt) if "DTI" in modes else None
        flair = self._get_flair(req.cached_flair, prompt) if "FLAIR" in modes else None

        technique = [m for m in modes if m in ALL_MODES]

        # Build findings
        findings: list[ReportFinding] = []
        if t1:
            findings.extend(self._analyze_t1(t1, model_info.n_params, loc))
        if t2:
            findings.extend(self._analyze_t2(t2, loc))
        if fmri:
            findings.extend(self._analyze_fmri(fmri, loc))
        if dti:
            findings.extend(self._analyze_dti(dti, loc))
        if flair:
            findings.extend(self._analyze_flair(flair, loc))

        # SAE findings (if provided)
        if req.cached_sae:
            sae_data = SAEData(**req.cached_sae)
            findings.extend(self._analyze_sae(sae_data, loc))
            if "SAE" not in technique:
                technique.append("SAE")

        # Battery findings (if provided)
        if req.cached_battery:
            findings.extend(self._analyze_battery(req.cached_battery, loc))
            if "battery" not in technique:
                technique.append("battery")

        impressions = self._build_impressions(findings, loc)
        recommendations = self._build_recommendations(findings, loc)

        elapsed_ms = (time.time() - start) * 1000
        logger.info("Report generated in %.1fms, %d findings", elapsed_ms, len(findings))

        return DiagnosticReport(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            total_params=model_info.n_params,
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            prompt=prompt,
            technique=technique,
            findings=findings,
            impressions=impressions,
            recommendations=recommendations,
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    # ------------------------------------------------------------------ #
    # Data retrieval helpers
    # ------------------------------------------------------------------ #

    def _get_t1(self, cached: dict | None) -> StructuralData:
        if cached:
            return StructuralData(**cached)
        return self._engine.scan_structural()

    def _get_t2(self, cached: dict | None) -> WeightData:
        if cached:
            return WeightData(**cached)
        return self._engine.scan_weights()

    def _get_fmri(self, cached: dict | None, prompt: str) -> ActivationData:
        if cached:
            return ActivationData(**cached)
        return self._engine.scan_activation(ActivationScanRequest(prompt=prompt))

    def _get_dti(self, cached: dict | None, prompt: str) -> CircuitData:
        if cached:
            return CircuitData(**cached)
        return self._engine.scan_circuits(CircuitScanRequest(prompt=prompt))

    def _get_flair(self, cached: dict | None, prompt: str) -> AnomalyData:
        if cached:
            return AnomalyData(**cached)
        return self._engine.scan_anomaly(AnomalyScanRequest(prompt=prompt))

    # ------------------------------------------------------------------ #
    # T1 Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_t1(data: StructuralData, total_params: int, loc: str) -> list[ReportFinding]:
        n_layers = sum(1 for ly in data.layers if ly.layer_type == "attention")
        n_mlp = sum(1 for ly in data.layers if ly.layer_type == "mlp")
        param_m = total_params / 1e6

        details = [
            T(loc, "report.arch_layers", n=n_layers),
            T(loc, "report.arch_params", m=param_m),
            T(
                loc,
                "report.arch_components",
                total=len(data.layers),
                attn=n_layers,
                mlp=n_mlp,
            ),
            T(loc, "report.arch_connections", n=len(data.connections)),
        ]

        severity = "normal"
        if n_layers < 4:
            severity = "notable"
            details.append(T(loc, "report.arch_shallow"))
            explanation = T(loc, "report.explain_t1_shallow", n=n_layers)
        elif n_layers > 40:
            severity = "notable"
            details.append(T(loc, "report.arch_deep"))
            explanation = T(loc, "report.explain_t1_deep", n=n_layers)
        else:
            explanation = T(
                loc,
                "report.explain_t1_normal",
                n=n_layers,
                params=param_m,
            )

        return [
            ReportFinding(
                scan_mode="T1",
                severity=severity,
                title=T(loc, "report.arch_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # T2 Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_t2(data: WeightData, loc: str) -> list[ReportFinding]:
        if not data.layers:
            return [
                ReportFinding(
                    scan_mode="T2",
                    severity="normal",
                    title=T(loc, "report.weight_title"),
                    details=[T(loc, "report.weight_no_data")],
                )
            ]

        all_l2 = [ly.l2_norm for ly in data.layers]
        all_std = [ly.std for ly in data.layers]
        avg_l2 = sum(all_l2) / len(all_l2)
        std_l2 = (
            (sum((x - avg_l2) ** 2 for x in all_l2) / len(all_l2)) ** 0.5 if len(all_l2) > 1 else 0
        )
        avg_std = sum(all_std) / len(all_std)

        details = [
            T(loc, "report.weight_scanned", n=len(data.layers)),
            T(loc, "report.weight_avg_l2", avg=avg_l2, std=std_l2),
            T(loc, "report.weight_avg_std", v=avg_std),
        ]

        severity = "normal"
        outlier_layers: list[str] = []
        for layer in data.layers:
            if layer.l2_norm > avg_l2 + 2 * std_l2 and std_l2 > 0:
                outlier_layers.append(f"{layer.layer_id}/{layer.component}")

        if outlier_layers:
            severity = "notable" if len(outlier_layers) <= 3 else "warning"
            details.append(T(loc, "report.weight_outlier", layers=", ".join(outlier_layers[:5])))

        near_zero = [ly for ly in data.layers if ly.l2_norm < 0.01]
        if near_zero:
            severity = "warning"
            detail_str = ", ".join(f"{nz.layer_id}/{nz.component}" for nz in near_zero[:3])
            details.append(T(loc, "report.weight_near_zero", layers=detail_str))

        if near_zero:
            explanation = T(loc, "report.explain_t2_near_zero")
        elif outlier_layers:
            explanation = T(
                loc,
                "report.explain_t2_outlier",
                threshold=avg_l2 + 2 * std_l2,
                layers=", ".join(outlier_layers[:3]),
            )
        else:
            explanation = T(loc, "report.explain_t2_normal", std=avg_std)

        return [
            ReportFinding(
                scan_mode="T2",
                severity=severity,
                title=T(loc, "report.weight_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # fMRI Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_fmri(data: ActivationData, loc: str) -> list[ReportFinding]:
        if not data.layers:
            return []

        last_tok_idx = len(data.tokens) - 1
        high_layers: list[str] = []
        for layer in data.layers:
            if last_tok_idx < len(layer.activations) and layer.activations[last_tok_idx] > 0.7:
                high_layers.append(layer.layer_id)

        details = [
            T(loc, "report.fmri_tokens", n=len(data.tokens)),
            T(loc, "report.fmri_layers", n=len(data.layers)),
        ]

        severity = "normal"
        if high_layers:
            severity = "notable"
            details.append(T(loc, "report.fmri_high", layers=", ".join(high_layers[:5])))
            attn_with_heads = [ly for ly in data.layers if ly.per_head is not None]
            if attn_with_heads:
                details.append(T(loc, "report.fmri_heads", n=len(attn_with_heads)))
            explanation = T(
                loc,
                "report.explain_fmri_high",
                layers=", ".join(high_layers[:5]),
            )
        else:
            details.append(T(loc, "report.fmri_no_high"))
            explanation = T(loc, "report.explain_fmri_normal")

        return [
            ReportFinding(
                scan_mode="fMRI",
                severity=severity,
                title=T(loc, "report.fmri_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # DTI Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_dti(data: CircuitData, loc: str) -> list[ReportFinding]:
        total_comps = len(data.components)
        pathway_comps = sum(1 for c in data.components if c.is_pathway)
        pathway_ratio = pathway_comps / total_comps if total_comps > 0 else 0

        details = [
            T(loc, "report.dti_components", n=total_comps),
            T(loc, "report.dti_pathways", n=pathway_comps, pct=pathway_ratio),
            T(
                loc,
                "report.dti_active",
                n=sum(1 for c in data.connections if c.is_pathway),
            ),
        ]

        severity = "normal"
        if pathway_ratio < 0.15:
            severity = "notable"
            details.append(T(loc, "report.dti_low_density"))
            explanation = T(loc, "report.explain_dti_sparse", pct=pathway_ratio)
        elif pathway_ratio > 0.30:
            severity = "notable"
            details.append(T(loc, "report.dti_high_density"))
            explanation = T(loc, "report.explain_dti_dense", pct=pathway_ratio)
        else:
            explanation = T(loc, "report.explain_dti_normal", pct=pathway_ratio)

        pathway_ids = [c.layer_id for c in data.components if c.is_pathway]
        if len(pathway_ids) >= 3:
            chain_str = " \u2192 ".join(pathway_ids[:5])
            details.append(T(loc, "report.dti_chain", chain=chain_str))

        return [
            ReportFinding(
                scan_mode="DTI",
                severity=severity,
                title=T(loc, "report.dti_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # FLAIR Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_flair(data: AnomalyData, loc: str) -> list[ReportFinding]:
        if not data.layers:
            return []

        all_scores: list[float] = []
        for layer in data.layers:
            all_scores.extend(layer.anomaly_scores)

        avg_anomaly = sum(all_scores) / len(all_scores) if all_scores else 0
        max_anomaly = max(all_scores) if all_scores else 0

        details = [
            T(loc, "report.flair_scope", layers=len(data.layers), tokens=len(data.tokens)),
            T(loc, "report.flair_avg", v=avg_anomaly),
            T(loc, "report.flair_peak", v=max_anomaly),
        ]

        severity = "normal"
        notable_count = sum(1 for s in all_scores if s > 0.6)
        warning_count = sum(1 for s in all_scores if s > 0.8)

        if warning_count > 0:
            severity = "warning"
            details.append(T(loc, "report.flair_warning", n=warning_count))
            explanation = T(loc, "report.explain_flair_warning", n=warning_count)
        elif notable_count > 0:
            severity = "notable"
            details.append(T(loc, "report.flair_notable", n=notable_count))
            explanation = T(loc, "report.explain_flair_notable", n=notable_count)
        else:
            details.append(T(loc, "report.flair_clean"))
            explanation = T(loc, "report.explain_flair_clean")

        return [
            ReportFinding(
                scan_mode="FLAIR",
                severity=severity,
                title=T(loc, "report.flair_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # Battery Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_battery(battery_data: dict, loc: str) -> list[ReportFinding]:
        total = battery_data.get("total_tests", 0)
        passed = battery_data.get("passed", 0)
        failed = battery_data.get("failed", 0)
        results = battery_data.get("results", [])

        details = [T(loc, "report.battery_passed", passed=passed, total=total)]

        severity = "normal"
        if failed > 0:
            severity = "warning" if failed >= total / 2 else "notable"
            failed_names = [r["name"] for r in results if not r.get("passed", True)]
            details.append(T(loc, "report.battery_failed", names=", ".join(failed_names[:5])))

        categories_failed: dict[str, int] = {}
        for r in results:
            if not r.get("passed", True):
                cat = r.get("category", "unknown")
                categories_failed[cat] = categories_failed.get(cat, 0) + 1
        for cat, count in categories_failed.items():
            details.append(T(loc, "report.battery_cat_fail", cat=cat, n=count))

        # SAE insights if present
        sae_summary = battery_data.get("sae_summary")
        if sae_summary:
            layer_idx = sae_summary.get("layer_idx", "?")
            n_unique = sae_summary.get("total_unique_features", 0)
            details.append(T(loc, "report.battery_sae_layer", layer=layer_idx, n_unique=n_unique))

            cross_features = sae_summary.get("cross_test_features", [])
            if cross_features:
                top = cross_features[0]
                details.append(T(
                    loc,
                    "report.battery_sae_cross",
                    idx=top["feature_idx"],
                    count=top["count"],
                    total=total,
                ))

        if failed > 0:
            cats_str = ", ".join(categories_failed.keys())
            explanation = T(
                loc,
                "report.explain_battery_fail",
                failed=failed,
                total=total,
                categories=cats_str,
            )
        else:
            explanation = T(loc, "report.explain_battery_pass", total=total)

        return [
            ReportFinding(
                scan_mode="battery",
                severity=severity,
                title=T(loc, "report.battery_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # SAE Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_sae(data: SAEData, loc: str) -> list[ReportFinding]:
        n_tokens = len(data.tokens)
        n_active = len(data.heatmap_feature_indices)

        details = [
            T(loc, "report.sae_layer", layer=data.layer_idx, d_sae=data.d_sae),
            T(loc, "report.sae_sparsity", pct=data.sparsity),
            T(loc, "report.sae_recon", loss=data.reconstruction_loss),
            T(loc, "report.sae_active", n=n_active, t=n_tokens),
        ]

        severity = "normal"
        explanation_key = "report.explain_sae_normal"
        explain_kwargs: dict = {
            "layer": data.layer_idx,
            "d_sae": data.d_sae,
            "pct": data.sparsity,
        }

        # Find the strongest feature across all tokens (skip BOS)
        max_act = 0.0
        max_feat_idx = 0
        max_token_str = ""
        second_max_act = 0.0

        for tf in data.token_features:
            if tf.top_features:
                top = tf.top_features[0]
                if top.activation > max_act:
                    second_max_act = max_act
                    max_act = top.activation
                    max_feat_idx = top.feature_idx
                    max_token_str = tf.token_str.strip() or " "
                elif top.activation > second_max_act:
                    second_max_act = top.activation

        # Report top feature per non-trivial token (skip BOS, limit to 3)
        reported = 0
        for tf in data.token_features:
            if reported >= 3:
                break
            tok_str = tf.token_str.strip()
            if not tok_str or tok_str in ("<|endoftext|>", "<s>", "<bos>"):
                continue
            if tf.top_features:
                top = tf.top_features[0]
                if top.activation > 0:
                    details.append(
                        T(
                            loc,
                            "report.sae_top_feature",
                            token=tok_str,
                            idx=top.feature_idx,
                            act=top.activation,
                        )
                    )
                    reported += 1

        # Check for dominant feature
        if max_act > 0 and second_max_act > 0:
            dominance_ratio = max_act / (max_act + second_max_act)
            if dominance_ratio > 0.7:
                severity = "notable"
                details.append(
                    T(
                        loc,
                        "report.sae_dominant",
                        idx=max_feat_idx,
                        ratio=dominance_ratio,
                    )
                )
                explanation_key = "report.explain_sae_dominant"
                explain_kwargs = {
                    "idx": max_feat_idx,
                    "token": max_token_str,
                    "ratio": dominance_ratio,
                }

        # Check reconstruction loss quality
        if data.reconstruction_loss > 1.0:
            severity = "notable"
            details.append(T(loc, "report.sae_high_recon"))
            explanation_key = "report.explain_sae_high_recon"
            explain_kwargs = {"loss": data.reconstruction_loss, "d_sae": data.d_sae}
        elif data.reconstruction_loss < 0.01:
            details.append(T(loc, "report.sae_low_recon"))

        # Neuronpedia availability
        has_np = any(
            f.neuronpedia_url
            for tf in data.token_features
            for f in tf.top_features
        )
        if has_np:
            details.append(T(loc, "report.sae_neuronpedia"))

        explanation = T(loc, explanation_key, **explain_kwargs)

        return [
            ReportFinding(
                scan_mode="SAE",
                severity=severity,
                title=T(loc, "report.sae_title"),
                details=details,
                explanation=explanation,
            )
        ]

    # ------------------------------------------------------------------ #
    # Impressions & Recommendations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_impressions(findings: list[ReportFinding], loc: str) -> list[ReportImpression]:
        impressions: list[ReportImpression] = []
        idx = 1

        warning_count = sum(1 for f in findings if f.severity == "warning")
        notable_count = sum(1 for f in findings if f.severity == "notable")
        normal_count = sum(1 for f in findings if f.severity == "normal")

        if warning_count == 0 and notable_count == 0:
            impressions.append(
                ReportImpression(
                    index=idx,
                    text=T(loc, "report.imp_healthy"),
                    severity="normal",
                )
            )
            idx += 1
        else:
            if warning_count > 0:
                modes = ", ".join(f.scan_mode for f in findings if f.severity == "warning")
                impressions.append(
                    ReportImpression(
                        index=idx,
                        text=T(loc, "report.imp_warning", modes=modes),
                        severity="warning",
                    )
                )
                idx += 1

            if notable_count > 0:
                modes = ", ".join(f.scan_mode for f in findings if f.severity == "notable")
                impressions.append(
                    ReportImpression(
                        index=idx,
                        text=T(loc, "report.imp_notable", modes=modes),
                        severity="notable",
                    )
                )
                idx += 1

        total = len(findings)
        if total > 0:
            impressions.append(
                ReportImpression(
                    index=idx,
                    text=T(
                        loc,
                        "report.imp_summary",
                        total=total,
                        normal=normal_count,
                        notable=notable_count,
                        warning=warning_count,
                    ),
                    severity="normal",
                )
            )

        return impressions

    @staticmethod
    def _build_recommendations(findings: list[ReportFinding], loc: str) -> list[str]:
        recs: list[str] = []

        for f in findings:
            if f.severity == "warning":
                if f.scan_mode == "T2":
                    recs.append(T(loc, "report.rec_weight"))
                elif f.scan_mode == "FLAIR":
                    recs.append(T(loc, "report.rec_flair"))
                elif f.scan_mode == "battery":
                    recs.append(T(loc, "report.rec_battery"))
            elif f.severity == "notable":
                if f.scan_mode == "fMRI":
                    recs.append(T(loc, "report.rec_fmri"))
                elif f.scan_mode == "DTI":
                    recs.append(T(loc, "report.rec_dti"))
                elif f.scan_mode == "SAE":
                    recs.append(T(loc, "report.rec_sae"))

        if not recs:
            recs.append(T(loc, "report.rec_default"))

        if len(recs) < 3:
            recs.append(T(loc, "report.rec_run_battery"))

        return recs
