"""Bilingual (en/ko) translation strings for report and battery engines."""

from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ #
    # Report Engine — T1
    # ------------------------------------------------------------------ #
    "report.arch_title": {
        "en": "Architecture",
        "ko": "\uc544\ud0a4\ud14d\ucc98",
    },
    "report.arch_layers": {
        "en": "{n}-layer transformer architecture",
        "ko": "{n}\ub808\uc774\uc5b4 \ud2b8\ub79c\uc2a4\ud3ec\uba38 \uc544\ud0a4\ud14d\ucc98",
    },
    "report.arch_params": {
        "en": "{m:.0f}M total parameters",
        "ko": "\ucd1d {m:.0f}M \ud30c\ub77c\ubbf8\ud130",
    },
    "report.arch_components": {
        "en": "{total} components ({attn} attention + {mlp} MLP + embed/unembed)",
        "ko": (
            "{total}\uac1c \ucef4\ud3ec\ub10c\ud2b8 "
            "({attn} \uc5b4\ud150\uc158 + {mlp} MLP + \uc784\ubca0\ub529/\uc5b8\uc784\ubca0\ub529)"
        ),
    },
    "report.arch_connections": {
        "en": "{n} sequential connections",
        "ko": "{n}\uac1c \uc21c\ucc28 \uc5f0\uacb0",
    },
    "report.arch_shallow": {
        "en": "Shallow architecture \u2014 limited representational depth",
        "ko": (
            "\uc5c5\uc740 \uc544\ud0a4\ud14d\ucc98 \u2014 "
            "\uc81c\ud55c\ub41c \ud45c\ud604 \uae4a\uc774"
        ),
    },
    "report.arch_deep": {
        "en": "Very deep architecture \u2014 may be prone to gradient issues",
        "ko": (
            "\ub9e4\uc6b0 \uae4a\uc740 \uc544\ud0a4\ud14d\ucc98 \u2014 "
            "\uae30\uc6b8\uae30 \ubb38\uc81c \uac00\ub2a5\uc131"
        ),
    },
    # ------------------------------------------------------------------ #
    # Report Engine — T2
    # ------------------------------------------------------------------ #
    "report.weight_title": {
        "en": "Weight Distribution",
        "ko": "\uac00\uc911\uce58 \ubd84\ud3ec",
    },
    "report.weight_no_data": {
        "en": "No weight data",
        "ko": "\uac00\uc911\uce58 \ub370\uc774\ud130 \uc5c6\uc74c",
    },
    "report.weight_scanned": {
        "en": "{n} weight tensors scanned",
        "ko": "{n}\uac1c \uac00\uc911\uce58 \ud150\uc11c \uc2a4\uce94",
    },
    "report.weight_avg_l2": {
        "en": "Average L2 norm: {avg:.2f} (std: {std:.2f})",
        "ko": "\ud3c9\uade0 L2 \ub178\ub984: {avg:.2f} (\ud45c\uc900\ud3b8\ucc28: {std:.2f})",
    },
    "report.weight_avg_std": {
        "en": "Average weight std: {v:.4f}",
        "ko": "\ud3c9\uade0 \uac00\uc911\uce58 \ud45c\uc900\ud3b8\ucc28: {v:.4f}",
    },
    "report.weight_outlier": {
        "en": "L2 norm outliers (>avg+2\u03c3): {layers}",
        "ko": "L2 \ub178\ub984 \uc774\uc0c1\uce58 (>avg+2\u03c3): {layers}",
    },
    "report.weight_near_zero": {
        "en": "Near-zero weight tensors: {layers}",
        "ko": "\uadfc\uc601 \uac00\uc911\uce58 \ud150\uc11c: {layers}",
    },
    # ------------------------------------------------------------------ #
    # Report Engine — fMRI
    # ------------------------------------------------------------------ #
    "report.fmri_title": {
        "en": "Activation Pattern",
        "ko": "\ud65c\uc131\ud654 \ud328\ud134",
    },
    "report.fmri_tokens": {
        "en": "{n} tokens processed",
        "ko": "{n}\uac1c \ud1a0\ud070 \ucc98\ub9ac",
    },
    "report.fmri_layers": {
        "en": "{n} layers analyzed",
        "ko": "{n}\uac1c \ub808\uc774\uc5b4 \ubd84\uc11d",
    },
    "report.fmri_high": {
        "en": "High activation (>0.7) at last token: {layers}",
        "ko": "\ub9c8\uc9c0\ub9c9 \ud1a0\ud070\uc5d0\uc11c \uace0\ud65c\uc131 (>0.7): {layers}",
    },
    "report.fmri_heads": {
        "en": "{n} attention layers with per-head data",
        "ko": (
            "{n}\uac1c \uc5b4\ud150\uc158 \ub808\uc774\uc5b4\uc5d0 "
            "\ud5e4\ub4dc\ubcc4 \ub370\uc774\ud130"
        ),
    },
    "report.fmri_no_high": {
        "en": "No high-activation layers detected at prediction position",
        "ko": (
            "\uc608\uce21 \uc704\uce58\uc5d0\uc11c "
            "\uace0\ud65c\uc131 \ub808\uc774\uc5b4 \ubbf8\ud0d0\uc9c0"
        ),
    },
    # ------------------------------------------------------------------ #
    # Report Engine — DTI
    # ------------------------------------------------------------------ #
    "report.dti_title": {
        "en": "Circuit Analysis",
        "ko": "\ud68c\ub85c \ubd84\uc11d",
    },
    "report.dti_components": {
        "en": "{n} components analyzed",
        "ko": "{n}\uac1c \ucef4\ud3ec\ub10c\ud2b8 \ubd84\uc11d",
    },
    "report.dti_pathways": {
        "en": "{n} pathway components ({pct:.0%} of total)",
        "ko": "{n}\uac1c \uacbd\ub85c \ucef4\ud3ec\ub10c\ud2b8 (\uc804\uccb4\uc758 {pct:.0%})",
    },
    "report.dti_active": {
        "en": "{n} active pathway connections",
        "ko": "{n}\uac1c \ud65c\uc131 \uacbd\ub85c \uc5f0\uacb0",
    },
    "report.dti_low_density": {
        "en": "Low pathway density \u2014 diffuse processing pattern",
        "ko": (
            "\ub0ae\uc740 \uacbd\ub85c \ubc00\ub3c4 \u2014 \ubd84\uc0b0 \ucc98\ub9ac \ud328\ud134"
        ),
    },
    "report.dti_high_density": {
        "en": "High pathway density \u2014 concentrated processing",
        "ko": "\ub192\uc740 \uacbd\ub85c \ubc00\ub3c4 \u2014 \uc9d1\uc911 \ucc98\ub9ac",
    },
    "report.dti_chain": {
        "en": "Critical pathway: {chain}",
        "ko": "\ud575\uc2ec \uacbd\ub85c: {chain}",
    },
    # ------------------------------------------------------------------ #
    # Report Engine — FLAIR
    # ------------------------------------------------------------------ #
    "report.flair_title": {
        "en": "Anomaly Detection",
        "ko": "\uc774\uc0c1 \ud0d0\uc9c0",
    },
    "report.flair_scope": {
        "en": "{layers} layers, {tokens} tokens",
        "ko": "{layers}\uac1c \ub808\uc774\uc5b4, {tokens}\uac1c \ud1a0\ud070",
    },
    "report.flair_avg": {
        "en": "Average anomaly score: {v:.3f}",
        "ko": "\ud3c9\uade0 \uc774\uc0c1 \uc810\uc218: {v:.3f}",
    },
    "report.flair_peak": {
        "en": "Peak anomaly: {v:.3f}",
        "ko": "\ucd5c\ub300 \uc774\uc0c1: {v:.3f}",
    },
    "report.flair_warning": {
        "en": "{n} high-anomaly regions (>0.8)",
        "ko": "{n}\uac1c \uace0\uc774\uc0c1 \uc601\uc5ed (>0.8)",
    },
    "report.flair_notable": {
        "en": "{n} elevated anomaly regions (>0.6)",
        "ko": "{n}\uac1c \uc0c1\uc2b9\ub41c \uc774\uc0c1 \uc601\uc5ed (>0.6)",
    },
    "report.flair_clean": {
        "en": "No significant anomalies detected",
        "ko": "\uc720\uc758\ubbf8\ud55c \uc774\uc0c1 \ubbf8\ud0d0\uc9c0",
    },
    # ------------------------------------------------------------------ #
    # Report Engine — Battery findings
    # ------------------------------------------------------------------ #
    "report.battery_title": {
        "en": "Functional Testing",
        "ko": "\uae30\ub2a5 \uac80\uc0ac",
    },
    "report.battery_passed": {
        "en": "{passed}/{total} tests passed",
        "ko": "{passed}/{total}\uac1c \ud14c\uc2a4\ud2b8 \ud1b5\uacfc",
    },
    "report.battery_failed": {
        "en": "Failed: {names}",
        "ko": "\uc2e4\ud328: {names}",
    },
    "report.battery_cat_fail": {
        "en": "{cat}: {n} failure(s)",
        "ko": "{cat}: {n}\uac74 \uc2e4\ud328",
    },
    # ------------------------------------------------------------------ #
    # Report Engine — Impressions
    # ------------------------------------------------------------------ #
    "report.imp_healthy": {
        "en": "All scans within normal parameters. Model appears healthy.",
        "ko": (
            "\ubaa8\ub4e0 \uc2a4\uce94\uc774 \uc815\uc0c1 \ubc94\uc704 \ub0b4. "
            "\ubaa8\ub378 \uc0c1\ud0dc \uc591\ud638."
        ),
    },
    "report.imp_warning": {
        "en": "Warning-level findings in: {modes}. Review recommended.",
        "ko": "\uacbd\uace0 \uc218\uc900 \uc18c\uacac: {modes}. \uac80\ud1a0 \uad8c\uc7a5.",
    },
    "report.imp_notable": {
        "en": "Notable findings in: {modes}. Within expected variation.",
        "ko": "\uc8fc\uc758 \uc18c\uacac: {modes}. \uc608\uc0c1 \ubcc0\ub3d9 \ubc94\uc704 \ub0b4.",
    },
    "report.imp_summary": {
        "en": (
            "{total} scan modalities completed: "
            "{normal} normal, {notable} notable, {warning} warning."
        ),
        "ko": (
            "{total}\uac1c \uc2a4\uce94 \ubaa8\ub4dc \uc644\ub8cc: "
            "{normal}\uac1c \uc815\uc0c1, {notable}\uac1c \uc8fc\uc758, "
            "{warning}\uac1c \uacbd\uace0."
        ),
    },
    # ------------------------------------------------------------------ #
    # Report Engine — Recommendations
    # ------------------------------------------------------------------ #
    "report.rec_weight": {
        "en": (
            "Investigate weight outliers \u2014 consider if model "
            "has been fine-tuned or partially trained."
        ),
        "ko": (
            "\uac00\uc911\uce58 \uc774\uc0c1\uce58 \uc870\uc0ac \u2014 "
            "\ud30c\uc778\ud29c\ub2dd \ub610\ub294 "
            "\ubd88\uc644\uc804 \ud559\uc2b5 \uc5ec\ubd80 \ud655\uc778 \ud544\uc694."
        ),
    },
    "report.rec_flair": {
        "en": (
            "High anomaly scores detected \u2014 test with diverse "
            "prompts to check for systematic issues."
        ),
        "ko": (
            "\ub192\uc740 \uc774\uc0c1 \uc810\uc218 \ud0d0\uc9c0 \u2014 "
            "\ub2e4\uc591\ud55c \ud504\ub86c\ud504\ud2b8\ub85c "
            "\uccb4\uacc4\uc801 \ubb38\uc81c \ud655\uc778 \ud544\uc694."
        ),
    },
    "report.rec_battery": {
        "en": (
            "Functional test failures \u2014 verify model capability "
            "on failed categories with additional prompts."
        ),
        "ko": (
            "\uae30\ub2a5 \uac80\uc0ac \uc2e4\ud328 \u2014 "
            "\uc2e4\ud328 \uce74\ud14c\uace0\ub9ac\uc5d0 \ub300\ud574 "
            "\ucd94\uac00 \ud504\ub86c\ud504\ud2b8\ub85c \uac80\uc99d \ud544\uc694."
        ),
    },
    "report.rec_fmri": {
        "en": (
            "High activation patterns detected \u2014 consider DTI scan for circuit-level analysis."
        ),
        "ko": (
            "\uace0\ud65c\uc131 \ud328\ud134 \ud0d0\uc9c0 \u2014 "
            "\ud68c\ub85c \uc218\uc900 \ubd84\uc11d\uc744 \uc704\ud574 "
            "DTI \uc2a4\uce94 \uad8c\uc7a5."
        ),
    },
    "report.rec_dti": {
        "en": ("Unusual pathway density \u2014 run perturbation tests on key pathway components."),
        "ko": (
            "\ube44\uc815\uc0c1\uc801 \uacbd\ub85c \ubc00\ub3c4 \u2014 "
            "\ud575\uc2ec \uacbd\ub85c \ucef4\ud3ec\ub10c\ud2b8\uc5d0 "
            "perturbation \ud14c\uc2a4\ud2b8 \uad8c\uc7a5."
        ),
    },
    "report.rec_default": {
        "en": ("No immediate concerns. Consider testing with adversarial or edge-case prompts."),
        "ko": (
            "\uc989\uac01\uc801 \uc6b0\ub824 \uc0ac\ud56d \uc5c6\uc74c. "
            "\uc801\ub300\uc801/\uc5e3\uc9c0 \ucf00\uc774\uc2a4 "
            "\ud504\ub86c\ud504\ud2b8\ub85c \ud14c\uc2a4\ud2b8 \uad8c\uc7a5."
        ),
    },
    "report.rec_run_battery": {
        "en": "Run functional test battery for standardized capability assessment.",
        "ko": (
            "\ud45c\uc900\ud654\ub41c \ub2a5\ub825 \ud3c9\uac00\ub97c \uc704\ud574 "
            "\uae30\ub2a5 \uac80\uc0ac \ubc30\ud130\ub9ac \uc2e4\ud589 \uad8c\uc7a5."
        ),
    },
    # ------------------------------------------------------------------ #
    # Battery Engine
    # ------------------------------------------------------------------ #
    "battery.summary_all_passed": {
        "en": "{passed}/{total} tests passed. All tests passed.",
        "ko": "{passed}/{total}\uac1c \ud14c\uc2a4\ud2b8 \ud1b5\uacfc. \uc804\uccb4 \ud1b5\uacfc.",
    },
    "battery.summary_some_failed": {
        "en": (
            "{passed}/{total} tests passed. "
            "{failed} test(s) failed \u2014 see individual results for details."
        ),
        "ko": (
            "{passed}/{total}\uac1c \ud14c\uc2a4\ud2b8 \ud1b5\uacfc. "
            "{failed}\uac1c \uc2e4\ud328 \u2014 "
            "\uac1c\ubcc4 \uacb0\uacfc\uc5d0\uc11c \uc138\ubd80 \uc0ac\ud56d \ud655\uc778."
        ),
    },
    "battery.found_expected": {
        "en": (
            "Expected token '{token}' found in top-3 "
            "(prob={prob:.1%}). Model shows strong {category} capability."
        ),
        "ko": (
            "\uc608\uc0c1 \ud1a0\ud070 '{token}' top-3\uc5d0\uc11c \ubc1c\uacac "
            "(\ud655\ub960={prob:.1%}). "
            "\ubaa8\ub378\uc774 \uac15\ud55c {category} \ub2a5\ub825\uc744 \ubcf4\uc784."
        ),
    },
    "battery.found_acceptable": {
        "en": "Expected token '{token}' has prob={prob:.1%}. Acceptable recall.",
        "ko": (
            "\uc608\uc0c1 \ud1a0\ud070 '{token}' \ud655\ub960={prob:.1%}. "
            "\uc218\uc6a9 \uac00\ub2a5\ud55c \ud68c\uc0c1."
        ),
    },
    "battery.not_found": {
        "en": (
            "Expected one of {tokens} in top-3 with prob > 5%, "
            "but top prediction was '{actual}' ({prob:.1%})."
        ),
        "ko": (
            "top-3\uc5d0\uc11c {tokens} \uc911 \ud558\ub098 "
            "(prob > 5%) \uc608\uc0c1\ud588\uc73c\ub098, "
            "\uc2e4\uc81c \ucd5c\uc0c1\uc704 \uc608\uce21\uc740 '{actual}' ({prob:.1%})."
        ),
    },
    "battery.unexpected_found": {
        "en": ("Unexpected token '{token}' found in top-1. Model failed to process negation."),
        "ko": (
            "\ube44\uc608\uc0c1 \ud1a0\ud070 '{token}' top-1\uc5d0\uc11c \ubc1c\uacac. "
            "\ubaa8\ub378\uc774 \ubd80\uc815 \ucc98\ub9ac\uc5d0 \uc2e4\ud328."
        ),
    },
    "battery.unexpected_found_topk": {
        "en": ("Unexpected token '{token}' found in top-3. Model failed {category} test."),
        "ko": (
            "\ube44\uc608\uc0c1 \ud1a0\ud070 '{token}' top-3\uc5d0\uc11c \ubc1c\uacac. "
            "{category} \ud14c\uc2a4\ud2b8 \uc2e4\ud328."
        ),
    },
    "battery.avoided_unexpected": {
        "en": (
            "Correctly avoided unexpected tokens {tokens}. Top prediction: '{actual}' ({prob:.1%})."
        ),
        "ko": (
            "\ube44\uc608\uc0c1 \ud1a0\ud070 {tokens} \uc815\uc0c1 \ud68c\ud53c. "
            "\ucd5c\uc0c1\uc704 \uc608\uce21: '{actual}' ({prob:.1%})."
        ),
    },
    "battery.top_prediction": {
        "en": "Top prediction: '{actual}' ({prob:.1%}).",
        "ko": "\ucd5c\uc0c1\uc704 \uc608\uce21: '{actual}' ({prob:.1%}).",
    },
    "battery.bias_result": {
        "en": ("Pronoun ratio he/she = {ratio:.1f}:1 (he={he:.1%}, she={she:.1%}).{comp} "),
        "ko": (
            "\ub300\uba85\uc0ac \ube44\uc728 he/she = {ratio:.1f}:1 "
            "(he={he:.1%}, she={she:.1%}).{comp} "
        ),
    },
    "battery.bias_ok": {
        "en": "Within acceptable range.",
        "ko": "\ud5c8\uc6a9 \ubc94\uc704 \ub0b4.",
    },
    "battery.bias_fail": {
        "en": "Significant gender bias detected.",
        "ko": "\uc720\uc758\ubbf8\ud55c \uc131\ubcc4 \ud3b8\ud5a5 \ud0d0\uc9c0.",
    },
    "battery.comp_info": {
        "en": " Comparison '{prompt}': he={he:.1%}, she={she:.1%}.",
        "ko": " \ube44\uad50 '{prompt}': he={he:.1%}, she={she:.1%}.",
    },
    # ------------------------------------------------------------------ #
    # Finding Explanations
    # ------------------------------------------------------------------ #
    "report.explain_t1_normal": {
        "en": (
            "A {n}-layer transformer with {params:.0f}M parameters is "
            "a standard-scale architecture. Each layer contains an "
            "attention head (captures token relationships) and an MLP "
            "(transforms representations). The sequential stacking "
            "allows progressively more abstract feature extraction."
        ),
        "ko": (
            "{n}\ub808\uc774\uc5b4, {params:.0f}M \ud30c\ub77c\ubbf8\ud130\ub294 "
            "\ud45c\uc900\uc801\uc778 \uaddc\ubaa8\uc785\ub2c8\ub2e4. "
            "\uac01 \ub808\uc774\uc5b4\ub294 \uc5b4\ud150\uc158 \ud5e4\ub4dc"
            "(\ud1a0\ud070 \uac04 \uad00\uacc4 \ud3ec\ucc29)\uc640 "
            "MLP(\ud45c\ud604 \ubcc0\ud658)\ub85c \uad6c\uc131\ub429\ub2c8\ub2e4. "
            "\ub808\uc774\uc5b4\uac00 \uc30d\uc77c\uc218\ub85d "
            "\ub354 \ucd94\uc0c1\uc801\uc778 \ud2b9\uc9d5\uc744 "
            "\ucd94\ucd9c\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_t1_shallow": {
        "en": (
            "With only {n} layers, this model has limited depth "
            "for building complex representations. Shallow models "
            "may struggle with multi-step reasoning or capturing "
            "long-range dependencies between tokens."
        ),
        "ko": (
            "{n}\ub808\uc774\uc5b4\ub9cc\uc73c\ub85c\ub294 "
            "\ubcf5\uc7a1\ud55c \ud45c\ud604\uc744 \uad6c\ucd95\ud558\uae30 "
            "\uc5b4\ub835\uc2b5\ub2c8\ub2e4. "
            "\uc58c\uc740 \ubaa8\ub378\uc740 \ub2e4\ub2e8\uacc4 \ucd94\ub860\uc774\ub098 "
            "\ud1a0\ud070 \uac04 \uc7a5\uac70\ub9ac \uc758\uc874\uc131 "
            "\ud3ec\ucc29\uc5d0 \ud55c\uacc4\uac00 \uc788\uc744 \uc218 "
            "\uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_t1_deep": {
        "en": (
            "With {n} layers, this is a very deep architecture. "
            "While deeper models can learn richer representations, "
            "they are more susceptible to gradient vanishing/exploding "
            "issues and may have redundant layers."
        ),
        "ko": (
            "{n}\ub808\uc774\uc5b4\ub294 \ub9e4\uc6b0 \uae4a\uc740 "
            "\uad6c\uc870\uc785\ub2c8\ub2e4. "
            "\uae4a\uc740 \ubaa8\ub378\uc740 \ud48d\ubd80\ud55c "
            "\ud45c\ud604\uc744 \ud559\uc2b5\ud560 \uc218 \uc788\uc9c0\ub9cc, "
            "\uae30\uc6b8\uae30 \uc18c\uc2e4/\ud3ed\ubc1c \ubb38\uc81c\uc5d0 "
            "\ucde8\uc57d\ud558\uace0 \uc911\ubcf5 \ub808\uc774\uc5b4\uac00 "
            "\uc788\uc744 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_t2_normal": {
        "en": (
            "Weight distribution is stable. L2 norms are consistent "
            "across layers, indicating uniform training. Average weight "
            "std of {std:.4f} suggests balanced parameter utilization."
        ),
        "ko": (
            "\uac00\uc911\uce58 \ubd84\ud3ec\uac00 \uc548\uc815\uc801\uc785\ub2c8\ub2e4. "
            "L2 \ub178\ub984\uc774 \ub808\uc774\uc5b4 \uac04 "
            "\uc77c\uad00\uc801\uc774\uba70, \uade0\uc77c\ud55c "
            "\ud559\uc2b5\uc744 \ub098\ud0c0\ub0c5\ub2c8\ub2e4. "
            "\ud3c9\uade0 \uac00\uc911\uce58 \ud45c\uc900\ud3b8\ucc28 "
            "{std:.4f}\ub294 \uade0\ud615 \uc7a1\ud78c "
            "\ud30c\ub77c\ubbf8\ud130 \ud65c\uc6a9\uc744 \uc2dc\uc0ac\ud569\ub2c8\ub2e4."
        ),
    },
    "report.explain_t2_outlier": {
        "en": (
            "Some layers have abnormally large weights (>{threshold:.0f}). "
            "This often indicates fine-tuning artifacts where certain "
            "layers were heavily updated, or initialization issues. "
            "The affected layers ({layers}) may dominate model behavior."
        ),
        "ko": (
            "\uc77c\ubd80 \ub808\uc774\uc5b4\uc758 \uac00\uc911\uce58\uac00 "
            "\ube44\uc815\uc0c1\uc801\uc73c\ub85c \ud07d\ub2c8\ub2e4 "
            "(>{threshold:.0f}). "
            "\ud30c\uc778\ud29c\ub2dd \uacfc\uc815\uc5d0\uc11c "
            "\ud2b9\uc815 \ub808\uc774\uc5b4\uac00 \uacfc\ub3c4\ud558\uac8c "
            "\uc5c5\ub370\uc774\ud2b8\ub418\uc5c8\uac70\ub098, "
            "\ucd08\uae30\ud654 \ubb38\uc81c\uc77c \uc218 \uc788\uc2b5\ub2c8\ub2e4. "
            "\ud574\ub2f9 \ub808\uc774\uc5b4({layers})\uac00 "
            "\ubaa8\ub378 \ub3d9\uc791\uc744 \uc9c0\ubc30\ud560 \uc218 "
            "\uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_t2_near_zero": {
        "en": (
            "Near-zero weight layers are effectively inactive. "
            "These layers contribute almost nothing to the model's "
            "computation, suggesting they were not utilized during "
            "training or were pruned."
        ),
        "ko": (
            "\uadfc\uc601 \uac00\uc911\uce58 \ub808\uc774\uc5b4\ub294 "
            "\uc0ac\uc2e4\uc0c1 \ube44\ud65c\uc131 \uc0c1\ud0dc\uc785\ub2c8\ub2e4. "
            "\ubaa8\ub378 \uc5f0\uc0b0\uc5d0 \uac70\uc758 \uae30\uc5ec\ud558\uc9c0 "
            "\uc54a\uc73c\uba70, \ud559\uc2b5 \uc911 \ud65c\uc6a9\ub418\uc9c0 "
            "\uc54a\uc558\uac70\ub098 \ud504\ub8e8\ub2dd\ub41c "
            "\uac83\uc77c \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_fmri_normal": {
        "en": (
            "No layers show particularly strong activation at the "
            "prediction position. The model processes this input "
            "with evenly distributed computation across layers, "
            "which is typical for straightforward inputs."
        ),
        "ko": (
            "\uc608\uce21 \uc704\uce58\uc5d0\uc11c \ud2b9\ubcc4\ud788 "
            "\uac15\ud55c \ud65c\uc131\ud654\ub97c \ubcf4\uc774\ub294 "
            "\ub808\uc774\uc5b4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. "
            "\ubaa8\ub378\uc774 \uc774 \uc785\ub825\uc744 "
            "\ub808\uc774\uc5b4 \uc804\uccb4\uc5d0 \uace0\ub974\uac8c "
            "\ubd84\uc0b0\ud558\uc5ec \ucc98\ub9ac\ud558\uba70, "
            "\ub2e8\uc21c\ud55c \uc785\ub825\uc5d0\uc11c "
            "\uc804\ud615\uc801\uc778 \ud328\ud134\uc785\ub2c8\ub2e4."
        ),
    },
    "report.explain_fmri_high": {
        "en": (
            "Layers {layers} show strong activation (>0.7) at the "
            "final token. These layers are heavily involved in "
            "computing the next-token prediction. High activation "
            "in specific layers suggests specialized processing "
            "for this type of input."
        ),
        "ko": (
            "{layers} \ub808\uc774\uc5b4\uac00 \ub9c8\uc9c0\ub9c9 "
            "\ud1a0\ud070\uc5d0\uc11c \uac15\ud55c \ud65c\uc131\ud654"
            "(>0.7)\ub97c \ubcf4\uc785\ub2c8\ub2e4. "
            "\uc774 \ub808\uc774\uc5b4\ub4e4\uc774 \ub2e4\uc74c \ud1a0\ud070 "
            "\uc608\uce21\uc5d0 \ud06c\uac8c \uad00\uc5ec\ud558\uba70, "
            "\ud2b9\uc815 \ub808\uc774\uc5b4\uc758 \uace0\ud65c\uc131\uc740 "
            "\uc774 \uc720\ud615\uc758 \uc785\ub825\uc5d0 \ub300\ud55c "
            "\ud2b9\ud654\ub41c \ucc98\ub9ac\ub97c \uc2dc\uc0ac\ud569\ub2c8\ub2e4."
        ),
    },
    "report.explain_dti_normal": {
        "en": (
            "Pathway density is in the normal range ({pct:.0%}). "
            "Processing is appropriately distributed across "
            "components, indicating healthy circuit utilization."
        ),
        "ko": (
            "\uacbd\ub85c \ubc00\ub3c4\uac00 \uc815\uc0c1 \ubc94\uc704"
            "({pct:.0%})\uc785\ub2c8\ub2e4. "
            "\ucc98\ub9ac\uac00 \ucef4\ud3ec\ub10c\ud2b8 \uc804\uccb4\uc5d0 "
            "\uc801\uc808\ud788 \ubd84\uc0b0\ub418\uc5b4 \uc788\uc73c\uba70, "
            "\uac74\uac15\ud55c \ud68c\ub85c \ud65c\uc6a9\uc744 "
            "\ub098\ud0c0\ub0c5\ub2c8\ub2e4."
        ),
    },
    "report.explain_dti_sparse": {
        "en": (
            "Low pathway density ({pct:.0%}) means only a few "
            "components are critical for the prediction. The model "
            "relies on a narrow computational path, which can mean "
            "efficient but potentially fragile processing."
        ),
        "ko": (
            "\ub0ae\uc740 \uacbd\ub85c \ubc00\ub3c4({pct:.0%})\ub294 "
            "\uc18c\uc218 \ucef4\ud3ec\ub10c\ud2b8\ub9cc \uc608\uce21\uc5d0 "
            "\uc911\uc694\ud558\ub2e4\ub294 \uc758\ubbf8\uc785\ub2c8\ub2e4. "
            "\ubaa8\ub378\uc774 \uc88b\uc740 \uc5f0\uc0b0 \uacbd\ub85c\uc5d0 "
            "\uc758\uc874\ud558\ubbc0\ub85c \ud6a8\uc728\uc801\uc774\uc9c0\ub9cc "
            "\ucde8\uc57d\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_dti_dense": {
        "en": (
            "High pathway density ({pct:.0%}) means many components "
            "contribute to the prediction. This distributed processing "
            "pattern indicates the model engages broadly with the "
            "input, which can improve robustness."
        ),
        "ko": (
            "\ub192\uc740 \uacbd\ub85c \ubc00\ub3c4({pct:.0%})\ub294 "
            "\ub9ce\uc740 \ucef4\ud3ec\ub10c\ud2b8\uac00 \uc608\uce21\uc5d0 "
            "\uae30\uc5ec\ud55c\ub2e4\ub294 \uc758\ubbf8\uc785\ub2c8\ub2e4. "
            "\uc774\ub7ec\ud55c \ubd84\uc0b0 \ucc98\ub9ac \ud328\ud134\uc740 "
            "\ubaa8\ub378\uc774 \uc785\ub825\uc744 "
            "\uad11\ubc94\uc704\ud558\uac8c \ucc98\ub9ac\ud558\uba70, "
            "\uacac\uace0\uc131\uc744 \ub192\uc77c \uc218 "
            "\uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_flair_clean": {
        "en": (
            "All anomaly scores are low. Each layer's intermediate "
            "prediction aligns well with the final output, indicating "
            "smooth convergence through the network. The model "
            "processes this input without internal conflicts."
        ),
        "ko": (
            "\ubaa8\ub4e0 \uc774\uc0c1 \uc810\uc218\uac00 "
            "\ub0ae\uc2b5\ub2c8\ub2e4. "
            "\uac01 \ub808\uc774\uc5b4\uc758 \uc911\uac04 \uc608\uce21\uc774 "
            "\ucd5c\uc885 \ucd9c\ub825\uacfc \uc798 \uc77c\uce58\ud558\uc5ec "
            "\ub124\ud2b8\uc6cc\ud06c \uc804\uccb4\uc5d0\uc11c "
            "\ub9e4\ub044\ub7ec\uc6b4 \uc218\ub834\uc744 \ubcf4\uc785\ub2c8\ub2e4. "
            "\ub0b4\ubd80 \ucda9\ub3cc \uc5c6\uc774 "
            "\uc785\ub825\uc744 \ucc98\ub9ac\ud569\ub2c8\ub2e4."
        ),
    },
    "report.explain_flair_notable": {
        "en": (
            "{n} regions show elevated anomaly scores (>0.6). "
            "This means intermediate predictions at those layers "
            "differ from the final output. Some divergence in early "
            "layers is expected, but persistent divergence in later "
            "layers may indicate complex internal processing."
        ),
        "ko": (
            "{n}\uac1c \uc601\uc5ed\uc5d0\uc11c \uc0c1\uc2b9\ub41c "
            "\uc774\uc0c1 \uc810\uc218(>0.6)\uac00 "
            "\uad00\ucc30\ub429\ub2c8\ub2e4. "
            "\ud574\ub2f9 \ub808\uc774\uc5b4\uc758 \uc911\uac04 "
            "\uc608\uce21\uc774 \ucd5c\uc885 \ucd9c\ub825\uacfc "
            "\ub2e4\ub985\ub2c8\ub2e4. "
            "\ucd08\ubc18 \ub808\uc774\uc5b4\uc758 \ubc1c\uc0b0\uc740 "
            "\uc815\uc0c1\uc774\uc9c0\ub9cc, \ud6c4\ubc18 \ub808\uc774\uc5b4\uc758 "
            "\uc9c0\uc18d\uc801 \ubc1c\uc0b0\uc740 \ubcf5\uc7a1\ud55c "
            "\ub0b4\ubd80 \ucc98\ub9ac\ub97c \ub098\ud0c0\ub0bc \uc218 "
            "\uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_flair_warning": {
        "en": (
            "{n} high-anomaly regions (>0.8) detected. These layers "
            "produce intermediate predictions that strongly disagree "
            "with the final output. This can indicate that the model "
            "dramatically changes its 'mind' at these depths, which "
            "may signal processing instability for this input."
        ),
        "ko": (
            "{n}\uac1c \uace0\uc774\uc0c1 \uc601\uc5ed(>0.8)\uc774 "
            "\ud0d0\uc9c0\ub418\uc5c8\uc2b5\ub2c8\ub2e4. "
            "\ud574\ub2f9 \ub808\uc774\uc5b4\uc758 \uc911\uac04 "
            "\uc608\uce21\uc774 \ucd5c\uc885 \ucd9c\ub825\uacfc "
            "\ud06c\uac8c \ub2e4\ub985\ub2c8\ub2e4. "
            "\ubaa8\ub378\uc774 \uc774 \uae4a\uc774\uc5d0\uc11c "
            "\uc608\uce21\uc744 \uae09\uaca9\ud788 \ubcc0\uacbd\ud558\uba70, "
            "\uc774 \uc785\ub825\uc5d0 \ub300\ud55c "
            "\ucc98\ub9ac \ubd88\uc548\uc815\uc131\uc744 "
            "\ub098\ud0c0\ub0bc \uc218 \uc788\uc2b5\ub2c8\ub2e4."
        ),
    },
    "report.explain_battery_pass": {
        "en": (
            "All {total} functional tests passed. The model "
            "demonstrates basic capabilities in factual recall, "
            "syntactic processing, negation handling, and reasoning."
        ),
        "ko": (
            "\uc804\uccb4 {total}\uac1c \uae30\ub2a5 \uac80\uc0ac\ub97c "
            "\ud1b5\uacfc\ud588\uc2b5\ub2c8\ub2e4. "
            "\ubaa8\ub378\uc774 \uc0ac\uc2e4 \uae30\uc5b5, "
            "\uad6c\ubb38 \ucc98\ub9ac, \ubd80\uc815 \ucc98\ub9ac, "
            "\ucd94\ub860 \ub4f1 \uae30\ubcf8 \ub2a5\ub825\uc744 "
            "\ubcf4\uc5ec\uc90d\ub2c8\ub2e4."
        ),
    },
    "report.explain_battery_fail": {
        "en": (
            "{failed}/{total} tests failed. Weaknesses detected in: "
            "{categories}. This means the model may produce incorrect "
            "predictions for these types of tasks. Consider testing "
            "with additional prompts in failed categories."
        ),
        "ko": (
            "{failed}/{total}\uac1c \uac80\uc0ac\uc5d0\uc11c "
            "\uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4. "
            "\uc57d\uc810 \uc601\uc5ed: {categories}. "
            "\uc774\ub7ec\ud55c \uc720\ud615\uc758 \uc791\uc5c5\uc5d0\uc11c "
            "\ubaa8\ub378\uc774 \uc798\ubabb\ub41c \uc608\uce21\uc744 "
            "\ub0bc \uc218 \uc788\uc2b5\ub2c8\ub2e4. "
            "\uc2e4\ud328 \uce74\ud14c\uace0\ub9ac\uc5d0 \ub300\ud574 "
            "\ucd94\uac00 \ud504\ub86c\ud504\ud2b8\ub85c "
            "\uac80\uc99d\ud574\ubcf4\uc138\uc694."
        ),
    },
}


def T(locale: str, key: str, **kwargs: object) -> str:  # noqa: N802
    """Look up a translated string by key and format with kwargs."""
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    template = entry.get(locale, entry.get("en", key))
    if kwargs:
        return template.format(**kwargs)
    return template
