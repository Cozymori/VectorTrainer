"use client";

import { useEffect, useState, useCallback } from "react";
import {
  getHookScript,
  getHookVersions,
  rollbackHook,
} from "@/lib/api";
import type { HookVersion } from "@/lib/types";
import { Button } from "@/components/ui/button";
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

export default function HookScriptPage() {
  const [source, setSource] = useState("");
  const [scriptPath, setScriptPath] = useState("");
  const [versions, setVersions] = useState<HookVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [rollingBack, setRollingBack] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [hookData, versionsData] = await Promise.all([
        getHookScript(),
        getHookVersions(),
      ]);
      setSource(hookData.source);
      setScriptPath(hookData.path);
      setVersions(versionsData);
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

  const handleRollback = async (versionId: string) => {
    setRollingBack(true);
    try {
      await rollbackHook(versionId);
      await fetchData();
    } catch (e) {
      console.error("Rollback failed:", e);
    } finally {
      setRollingBack(false);
    }
  };

  if (loading) return <div className="text-muted-foreground">Loading...</div>;

  if (!source) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <p className="text-muted-foreground">
          Run the pipeline first to generate a hook script.
        </p>
      </div>
    );
  }

  const lines = source.split("\n");

  return (
    <div className="space-y-8">
      {/* Code viewer */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            Generated Hook Script
          </h2>
          <span className="text-xs text-muted-foreground font-mono">
            {scriptPath}
          </span>
        </div>
        <Card className="bg-[#0c0c0e] border-border overflow-hidden">
          <CardContent className="p-0">
            <div className="overflow-auto max-h-[500px]">
              <pre className="text-sm font-mono leading-relaxed">
                {lines.map((line, idx) => (
                  <div key={idx} className="flex hover:bg-white/[0.02]">
                    <span className="w-12 text-right pr-4 text-muted-foreground/50 select-none text-xs leading-relaxed">
                      {idx + 1}
                    </span>
                    <code className="flex-1 pr-4">{line || " "}</code>
                  </div>
                ))}
              </pre>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Version management */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Version History
        </h2>
        {versions.length > 0 ? (
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead className="text-right">Rules</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {versions.map((v) => (
                  <TableRow key={v.version_id}>
                    <TableCell className="font-mono text-sm text-primary">
                      {v.version_id}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {v.timestamp}
                    </TableCell>
                    <TableCell className="text-right">
                      {v.rules_count}
                    </TableCell>
                    <TableCell>
                      {v.active && (
                        <Badge className="bg-primary/20 text-primary border-primary/30">
                          Active
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      {!v.active && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={rollingBack}
                          onClick={() => handleRollback(v.version_id)}
                        >
                          Rollback
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No versions saved.</p>
        )}
      </section>
    </div>
  );
}
