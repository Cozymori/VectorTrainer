"use client";

import { useEffect, useState, useCallback } from "react";
import {
  getCostEstimate,
  getStats,
  startTraining,
  getTrainingStatus,
} from "@/lib/api";
import type { StatsResponse, CostEstimateResponse } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";

const MODELS = [
  "gpt-4o-mini-2024-07-18",
  "gpt-4o-2024-08-06",
  "gpt-3.5-turbo-0125",
];

export default function FineTuningPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [model, setModel] = useState(MODELS[0]);
  const [nEpochs, setNEpochs] = useState(3);
  const [apiKey, setApiKey] = useState("");
  const [maxBudget, setMaxBudget] = useState(10);
  const [costEst, setCostEst] = useState<CostEstimateResponse | null>(null);
  const [jobId, setJobId] = useState("");
  const [jobStatus, setJobStatus] = useState<Record<string, unknown> | null>(
    null
  );
  const [training, setTraining] = useState(false);
  const [checkingStatus, setCheckingStatus] = useState(false);
  const [error, setError] = useState("");

  const fetchStats = useCallback(async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const handler = () => fetchStats();
    window.addEventListener("pipeline-updated", handler);
    return () => window.removeEventListener("pipeline-updated", handler);
  }, [fetchStats]);

  const handleEstimate = async () => {
    try {
      const est = await getCostEstimate({ model, n_epochs: nEpochs });
      setCostEst(est);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cost estimation failed");
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setError("");
    try {
      const result = await startTraining({
        api_key: apiKey,
        model,
        n_epochs: nEpochs,
        max_budget_usd: maxBudget,
      });
      setJobId(result.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setTraining(false);
    }
  };

  const handleCheckStatus = async () => {
    if (!jobId) return;
    setCheckingStatus(true);
    try {
      const status = await getTrainingStatus(jobId);
      setJobStatus(status);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Status check failed");
    } finally {
      setCheckingStatus(false);
    }
  };

  if (!stats?.has_results) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <p className="text-muted-foreground">
          Run the pipeline first to generate training data.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Cost validation */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Cost Validation
        </h2>
        <div className="grid grid-cols-2 gap-6">
          <Card className="bg-card border-border">
            <CardContent className="p-5 space-y-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-2">
                  Model
                </label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MODELS.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-2">
                  Epochs: {nEpochs}
                </label>
                <Slider
                  value={[nEpochs]}
                  onValueChange={([v]) => setNEpochs(v)}
                  min={1}
                  max={10}
                  step={1}
                />
              </div>
              <Button
                variant="outline"
                className="w-full"
                onClick={handleEstimate}
              >
                Estimate Cost
              </Button>
            </CardContent>
          </Card>
          <div>
            {costEst && (
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  label="Tokens"
                  value={costEst.token_count.toLocaleString()}
                />
                <MetricCard
                  label="Estimated Cost"
                  value={`$${costEst.cost_estimate.toFixed(6)}`}
                />
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Training */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Start Training
        </h2>
        <div className="grid grid-cols-2 gap-6">
          <Card className="bg-card border-border">
            <CardContent className="p-5 space-y-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-2">
                  OpenAI API Key
                </label>
                <Input
                  type="password"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-2">
                  Max Budget (USD): ${maxBudget}
                </label>
                <Slider
                  value={[maxBudget]}
                  onValueChange={([v]) => setMaxBudget(v)}
                  min={1}
                  max={100}
                  step={1}
                />
              </div>
              <Button
                className="w-full"
                onClick={handleTrain}
                disabled={!apiKey || training}
              >
                {training ? "Starting..." : "Upload & Start Training"}
              </Button>
            </CardContent>
          </Card>

          {/* Status check */}
          <Card className="bg-card border-border">
            <CardContent className="p-5 space-y-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-2">
                  Job ID
                </label>
                <Input
                  placeholder="ftjob-..."
                  value={jobId}
                  onChange={(e) => setJobId(e.target.value)}
                />
              </div>
              <Button
                variant="outline"
                className="w-full"
                onClick={handleCheckStatus}
                disabled={!jobId || checkingStatus}
              >
                {checkingStatus ? "Checking..." : "Check Status"}
              </Button>
              {jobStatus && (
                <pre className="text-xs font-mono bg-muted/50 p-3 rounded-md overflow-auto max-h-48">
                  {JSON.stringify(jobStatus, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {error && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
