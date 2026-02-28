from __future__ import annotations

from pydantic import BaseModel


class ReportFinding(BaseModel):
    scan_mode: str  # "T1" | "T2" | "fMRI" | "DTI" | "FLAIR" | "battery"
    severity: str  # "normal" | "notable" | "warning"
    title: str
    details: list[str]
    explanation: str = ""


class ReportImpression(BaseModel):
    index: int
    text: str
    severity: str  # "normal" | "notable" | "warning"


class DiagnosticReport(BaseModel):
    model_id: str
    model_name: str
    total_params: int
    date: str
    prompt: str
    technique: list[str]
    findings: list[ReportFinding]
    impressions: list[ReportImpression]
    recommendations: list[str]
    metadata: dict


class ReportRequest(BaseModel):
    prompt: str | None = None
    include_modes: list[str] | None = None  # None = all modes
    cached_t1: dict | None = None
    cached_t2: dict | None = None
    cached_fmri: dict | None = None
    cached_dti: dict | None = None
    cached_flair: dict | None = None
    cached_battery: dict | None = None
    cached_sae: dict | None = None
    locale: str = "en"
