export interface StatsResponse {
  has_results: boolean;
  total_time: number | null;
  stages_completed: number;
  golden_candidates_count: number;
  rules_count: number;
  token_count: number;
  cost_1_epoch: number;
  cost_3_epoch: number;
  stage_times: Record<string, number>;
  per_function_stats: PerFunctionStat[];
  output_dir: string | null;
  jsonl_path: string | null;
  hook_script_path: string | null;
  mock_model: string | null;
  mock_job_id: string | null;
  function_name: string | null;
  epsilon: number | null;
  top_k: number | null;
}

export interface PerFunctionStat {
  function: string;
  steady: number;
  anomaly: number;
}

export interface GoldenCandidate {
  uuid: string;
  strategy: string;
  density: number;
  distance_to_centroid: number;
  return_value: string;
}

export interface Rule {
  rule_id: string;
  description: string;
  condition: string;
  action: string;
  priority: number;
  source_pair_index: number;
}

export interface DiffSegment {
  type: "added" | "removed" | "changed";
  bad: string;
  fixed: string;
}

export interface DiffAnalysis {
  edit_distance: number;
  similarity_score: number;
  diff_summary: string;
  diff_segments: DiffSegment[];
  input_prompt: string;
  context: Record<string, unknown>;
  bad_output: string;
  fixed_output: string;
}

export interface HookVersion {
  version_id: string;
  timestamp: string;
  rules_count: number;
  active: boolean;
  filename?: string;
}

export interface CostEstimateResponse {
  token_count: number;
  cost_estimate: number;
  model: string;
  n_epochs: number;
}

export interface PipelineRunResponse {
  status: string;
  total_time: number;
  stage_times: Record<string, number>;
  steady_count: number;
  anomaly_count: number;
  rules_count: number;
  token_count: number;
  cost_1_epoch: number;
  cost_3_epoch: number;
  per_function_stats: PerFunctionStat[];
}
