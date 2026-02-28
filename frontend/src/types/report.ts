export interface ReportFinding {
  scan_mode: string;
  severity: 'normal' | 'notable' | 'warning';
  title: string;
  details: string[];
  explanation: string;
}

export interface ReportImpression {
  index: number;
  text: string;
  severity: 'normal' | 'notable' | 'warning';
}

export interface DiagnosticReport {
  model_id: string;
  model_name: string;
  total_params: number;
  date: string;
  prompt: string;
  technique: string[];
  findings: ReportFinding[];
  impressions: ReportImpression[];
  recommendations: string[];
  metadata: Record<string, unknown>;
}

export interface ReportRequest {
  prompt?: string;
  locale?: string;
  include_modes?: string[];
  cached_t1?: Record<string, unknown>;
  cached_t2?: Record<string, unknown>;
  cached_fmri?: Record<string, unknown>;
  cached_dti?: Record<string, unknown>;
  cached_flair?: Record<string, unknown>;
  cached_battery?: Record<string, unknown>;
  cached_sae?: Record<string, unknown>;
}
