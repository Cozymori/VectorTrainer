const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function runPipeline(params: {
  epsilon: number;
  top_k: number;
  use_real: boolean;
  function_name: string;
}) {
  return apiFetch("/pipeline/run", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function getStats() {
  return apiFetch<import("./types").StatsResponse>("/stats");
}

export async function getGoldenData() {
  return apiFetch<import("./types").GoldenCandidate[]>("/golden-data");
}

export async function downloadGoldenData(): Promise<Blob> {
  const res = await fetch(`${API_BASE}/golden-data/download`);
  if (!res.ok) throw new Error("Download failed");
  return res.blob();
}

export async function getRules() {
  return apiFetch<{
    rules: import("./types").Rule[];
    diffs: import("./types").DiffAnalysis[];
  }>("/rules");
}

export async function getHookScript() {
  return apiFetch<{ source: string; path: string }>("/hook-script");
}

export async function getHookVersions() {
  return apiFetch<import("./types").HookVersion[]>("/hook-versions");
}

export async function rollbackHook(version_id: string) {
  return apiFetch<{ status: string; active_version: string }>("/hook-rollback", {
    method: "POST",
    body: JSON.stringify({ version_id }),
  });
}

export async function getCostEstimate(params: {
  model: string;
  n_epochs: number;
}) {
  return apiFetch<import("./types").CostEstimateResponse>("/cost-estimate", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function startTraining(params: {
  api_key: string;
  model: string;
  n_epochs: number;
  max_budget_usd: number;
}) {
  return apiFetch<{ status: string; file_id: string; job_id: string }>(
    "/pipeline/train",
    {
      method: "POST",
      body: JSON.stringify(params),
    }
  );
}

export async function getTrainingStatus(jobId: string) {
  return apiFetch<Record<string, unknown>>(`/pipeline/status/${jobId}`);
}

export async function getFeedbackPairs() {
  return apiFetch<import("./types").FeedbackPair[]>("/feedback-pairs");
}

export async function createFeedbackPair(data: {
  input_prompt: string;
  bad_output: string;
  fixed_output: string;
  context?: Record<string, unknown>;
}) {
  return apiFetch<import("./types").FeedbackPair>("/feedback-pairs", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateFeedbackPair(
  id: string,
  data: {
    input_prompt?: string;
    bad_output?: string;
    fixed_output?: string;
    context?: Record<string, unknown>;
  }
) {
  return apiFetch<import("./types").FeedbackPair>(`/feedback-pairs/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteFeedbackPair(id: string) {
  return apiFetch<{ status: string; deleted: string }>(
    `/feedback-pairs/${id}`,
    { method: "DELETE" }
  );
}
