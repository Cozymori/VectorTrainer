"use client";

import { useEffect, useState, useCallback } from "react";
import { getGoldenData, downloadGoldenData, getStats } from "@/lib/api";
import type { GoldenCandidate, StatsResponse } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function GoldenDataPage() {
  const [candidates, setCandidates] = useState<GoldenCandidate[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [data, st] = await Promise.all([getGoldenData(), getStats()]);
      setCandidates(data);
      setStats(st);
    } catch {
      // API may not be ready
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const handler = () => fetchData();
    window.addEventListener("pipeline-updated", handler);
    return () => window.removeEventListener("pipeline-updated", handler);
  }, [fetchData]);

  if (loading) {
    return <div className="text-muted-foreground">Loading...</div>;
  }

  if (!stats?.has_results) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <p className="text-muted-foreground">
          Run the pipeline first to see golden data results.
        </p>
      </div>
    );
  }

  const steadyCount = candidates.filter((c) => c.strategy === "steady").length;
  const anomalyCount = candidates.filter(
    (c) => c.strategy === "anomaly"
  ).length;

  // Density histogram
  const densities = candidates.map((c) => c.density);
  const minD = Math.min(...densities, 0);
  const maxD = Math.max(...densities, 1);
  const binCount = 12;
  const binSize = (maxD - minD) / binCount || 1;
  const bins = Array.from({ length: binCount }, (_, i) => {
    const lo = minD + i * binSize;
    const hi = lo + binSize;
    const count = densities.filter((d) => d >= lo && (i === binCount - 1 ? d <= hi : d < hi)).length;
    return { lo, hi, count };
  });
  const maxBinCount = Math.max(...bins.map((b) => b.count), 1);

  const handleDownload = async () => {
    try {
      const blob = await downloadGoldenData();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "golden_data.jsonl";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Download failed:", e);
    }
  };

  return (
    <div className="space-y-8">
      {/* Summary metrics */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Selection Summary
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <MetricCard label="Steady State" value={steadyCount} />
          <MetricCard label="Anomaly" value={anomalyCount} />
          <MetricCard label="Total" value={candidates.length} />
        </div>
      </section>

      {/* Candidates table */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            Golden Candidates
          </h2>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            Download JSONL ({candidates.length})
          </Button>
        </div>
        <div className="rounded-lg border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>UUID</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead className="text-right">Density</TableHead>
                <TableHead className="text-right">Distance</TableHead>
                <TableHead>Return Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {candidates.map((c) => (
                <TableRow key={c.uuid}>
                  <TableCell className="font-mono text-xs">{c.uuid}</TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        c.strategy === "steady" ? "default" : "secondary"
                      }
                      className={
                        c.strategy === "steady"
                          ? "bg-primary/20 text-primary border-primary/30"
                          : "bg-orange-500/20 text-orange-400 border-orange-500/30"
                      }
                    >
                      {c.strategy}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {c.density.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {c.distance_to_centroid.toFixed(4)}
                  </TableCell>
                  <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
                    {c.return_value}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </section>

      {/* Density histogram */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Density Distribution
        </h2>
        <div className="flex items-end gap-1 h-40">
          {bins.map((bin, idx) => (
            <div
              key={idx}
              className="flex-1 flex flex-col items-center justify-end h-full"
            >
              <div
                className="w-full rounded-t"
                style={{
                  height: `${(bin.count / maxBinCount) * 100}%`,
                  minHeight: bin.count > 0 ? "4px" : "0",
                  background:
                    "linear-gradient(180deg, #FF6500, #cc5100)",
                }}
              />
              <span className="text-[9px] text-muted-foreground mt-1 font-mono">
                {bin.lo.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
