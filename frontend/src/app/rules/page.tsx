"use client";

import { useEffect, useState, useCallback } from "react";
import { getRules } from "@/lib/api";
import type { Rule, DiffAnalysis } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function RulesPage() {
  const [rules, setRules] = useState<Rule[]>([]);
  const [diffs, setDiffs] = useState<DiffAnalysis[]>([]);
  const [expandedRule, setExpandedRule] = useState<string | null>(null);
  const [expandedDiff, setExpandedDiff] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const data = await getRules();
      setRules(data.rules);
      setDiffs(data.diffs);
    } catch {
      // ignore
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

  if (loading) return <div className="text-muted-foreground">Loading...</div>;

  if (rules.length === 0) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <p className="text-muted-foreground">
          Run the pipeline first to see synthesized rules.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Rules table */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Synthesized Rules
        </h2>
        <div className="rounded-lg border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-24">Rule ID</TableHead>
                <TableHead className="w-20">Priority</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="w-20">Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.map((r) => (
                <TableRow
                  key={r.rule_id}
                  className="cursor-pointer hover:bg-muted/30"
                  onClick={() =>
                    setExpandedRule(expandedRule === r.rule_id ? null : r.rule_id)
                  }
                >
                  <TableCell className="font-mono text-sm text-primary">
                    {r.rule_id}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{r.priority}</Badge>
                  </TableCell>
                  <TableCell className="text-sm">
                    {r.description.length > 100
                      ? r.description.slice(0, 100) + "..."
                      : r.description}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    Pair #{r.source_pair_index}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </section>

      {/* Expanded rule detail */}
      {expandedRule && (
        <section>
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
            Rule Detail: {expandedRule}
          </h2>
          {rules
            .filter((r) => r.rule_id === expandedRule)
            .map((r) => (
              <Card key={r.rule_id} className="bg-card border-border">
                <CardContent className="p-5 space-y-4">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase mb-1">
                      Description
                    </p>
                    <p className="text-sm">{r.description}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase mb-1">
                      Condition
                    </p>
                    <pre className="text-sm font-mono bg-muted/50 p-3 rounded-md overflow-auto">
                      {r.condition}
                    </pre>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase mb-1">
                      Action
                    </p>
                    <pre className="text-sm font-mono bg-muted/50 p-3 rounded-md overflow-auto">
                      {r.action}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
        </section>
      )}

      {/* Diff analysis */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Feedback Diff Analysis
        </h2>
        <div className="space-y-3">
          {diffs.map((diff, idx) => (
            <Card
              key={idx}
              className="bg-card border-border cursor-pointer hover:border-primary/30 transition-colors"
              onClick={() =>
                setExpandedDiff(expandedDiff === idx ? null : idx)
              }
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">
                    Diff #{idx + 1} — edit distance:{" "}
                    <span className="font-mono text-primary">
                      {diff.edit_distance}
                    </span>
                    , similarity:{" "}
                    <span className="font-mono text-primary">
                      {diff.similarity_score.toFixed(3)}
                    </span>
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {expandedDiff === idx ? "collapse" : "expand"}
                  </span>
                </div>

                {expandedDiff === idx && (
                  <div className="mt-4 space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground uppercase mb-1">
                          Bad Output
                        </p>
                        <pre className="text-sm font-mono bg-red-500/5 border border-red-500/20 p-3 rounded-md text-red-400">
                          {diff.bad_output}
                        </pre>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground uppercase mb-1">
                          Fixed Output
                        </p>
                        <pre className="text-sm font-mono bg-green-500/5 border border-green-500/20 p-3 rounded-md text-green-400 whitespace-pre-wrap">
                          {diff.fixed_output}
                        </pre>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground uppercase mb-1">
                        Summary
                      </p>
                      <p className="text-sm">{diff.diff_summary}</p>
                    </div>
                    {diff.diff_segments.length > 0 && (
                      <div>
                        <p className="text-xs text-muted-foreground uppercase mb-1">
                          Change Segments
                        </p>
                        <div className="space-y-1">
                          {diff.diff_segments.map((seg, si) => (
                            <div
                              key={si}
                              className="text-xs font-mono flex gap-2"
                            >
                              <Badge
                                variant="outline"
                                className={
                                  seg.type === "added"
                                    ? "text-green-400 border-green-500/30"
                                    : seg.type === "removed"
                                      ? "text-red-400 border-red-500/30"
                                      : "text-orange-400 border-orange-500/30"
                                }
                              >
                                {seg.type}
                              </Badge>
                              <span className="text-muted-foreground truncate">
                                {seg.type === "added"
                                  ? seg.fixed
                                  : seg.type === "removed"
                                    ? seg.bad
                                    : `${seg.bad.slice(0, 30)} → ${seg.fixed.slice(0, 30)}`}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
