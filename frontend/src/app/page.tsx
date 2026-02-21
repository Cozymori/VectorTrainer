"use client";

import { useEffect, useState, useCallback } from "react";
import { getStats } from "@/lib/api";
import type { StatsResponse } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { StageFlow } from "@/components/stage-flow";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const STAGE_INFO = [
  { key: "추출", description: "벡터 그래프 기반 밀도 분석" },
  { key: "합성", description: "피드백 차이 분석 및 규칙 합성" },
  { key: "준비", description: "훅 스크립트 생성" },
  { key: "학습", description: "파인튜닝 작업 실행" },
];

export default function PipelinePage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = useCallback(async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch {
      // API may not be ready
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const handler = () => fetchStats();
    window.addEventListener("pipeline-updated", handler);
    return () => window.removeEventListener("pipeline-updated", handler);
  }, [fetchStats]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!stats?.has_results) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center space-y-3">
          <p className="text-muted-foreground text-lg">
            No pipeline results yet.
          </p>
          <p className="text-muted-foreground text-sm">
            Set parameters in the sidebar and click <strong>Run Pipeline</strong>.
          </p>
        </div>
      </div>
    );
  }

  const stages = STAGE_INFO.map((s) => ({
    name: s.key,
    description: s.description,
    time: stats.stage_times[s.key] ?? 0,
  }));

  return (
    <div className="space-y-8">
      {/* Overview metrics */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Overview
        </h2>
        <div className="grid grid-cols-4 gap-4">
          <MetricCard
            label="Total Time"
            value={`${stats.total_time?.toFixed(3) ?? 0}s`}
          />
          <MetricCard
            label="Stages Completed"
            value={`${stats.stages_completed} / 4`}
          />
          <MetricCard
            label="Golden Candidates"
            value={stats.golden_candidates_count}
          />
          <MetricCard label="Rules" value={stats.rules_count} />
        </div>
      </section>

      {/* Cost metrics */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Cost Estimation
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <MetricCard
            label="Training Tokens"
            value={stats.token_count.toLocaleString()}
          />
          <MetricCard
            label="Cost (1 epoch)"
            value={`$${stats.cost_1_epoch.toFixed(6)}`}
          />
          <MetricCard
            label="Cost (3 epochs)"
            value={`$${stats.cost_3_epoch.toFixed(6)}`}
          />
        </div>
      </section>

      {/* Stage flow */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Pipeline Stages
        </h2>
        <StageFlow stages={stages} />
      </section>

      {/* Timing bar chart */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Stage Timing
        </h2>
        <div className="space-y-2">
          {stages.map((stage) => {
            const maxTime = Math.max(...stages.map((s) => s.time), 0.001);
            const pct = (stage.time / maxTime) * 100;
            return (
              <div key={stage.name} className="flex items-center gap-3">
                <span className="w-12 text-xs text-muted-foreground text-right font-mono">
                  {stage.name}
                </span>
                <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                  <div
                    className="h-full rounded flex items-center px-2"
                    style={{
                      width: `${Math.max(pct, 2)}%`,
                      background:
                        "linear-gradient(90deg, #FF6500, #cc5100)",
                    }}
                  >
                    <span className="text-xs font-mono text-background">
                      {stage.time.toFixed(3)}s
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Per-function stats */}
      {stats.per_function_stats.length > 1 && (
        <section>
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
            Per-function Extraction
          </h2>
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Function</TableHead>
                  <TableHead className="text-right">Steady</TableHead>
                  <TableHead className="text-right">Anomaly</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {stats.per_function_stats.map((s) => (
                  <TableRow key={s.function}>
                    <TableCell className="font-mono text-sm">
                      {s.function}
                    </TableCell>
                    <TableCell className="text-right">{s.steady}</TableCell>
                    <TableCell className="text-right">{s.anomaly}</TableCell>
                    <TableCell className="text-right font-semibold">
                      {s.steady + s.anomaly}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </section>
      )}
    </div>
  );
}
