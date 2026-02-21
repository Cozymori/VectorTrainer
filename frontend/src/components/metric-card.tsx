"use client";

import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
}

export function MetricCard({ label, value, sublabel }: MetricCardProps) {
  return (
    <Card className="bg-card border-border">
      <CardContent className="p-4">
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
          {label}
        </p>
        <p className="text-2xl font-bold text-gradient-cyan">{value}</p>
        {sublabel && (
          <p className="text-xs text-muted-foreground mt-1">{sublabel}</p>
        )}
      </CardContent>
    </Card>
  );
}
