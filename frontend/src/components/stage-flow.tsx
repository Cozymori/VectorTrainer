"use client";

interface Stage {
  name: string;
  description: string;
  time: number;
}

interface StageFlowProps {
  stages: Stage[];
}

export function StageFlow({ stages }: StageFlowProps) {
  const maxTime = Math.max(...stages.map((s) => s.time), 0.001);

  return (
    <div className="flex gap-3 w-full">
      {stages.map((stage, idx) => (
        <div key={stage.name} className="flex-1 flex flex-col items-center">
          <div className="flex items-center w-full mb-2">
            <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center text-xs font-mono text-primary">
              {idx + 1}
            </div>
            {idx < stages.length - 1 && (
              <div className="flex-1 h-px bg-primary/20 mx-2" />
            )}
          </div>
          <div className="w-full rounded-lg bg-card border border-border p-3 text-center">
            <p className="text-sm font-semibold mb-1">{stage.name}</p>
            <p className="text-lg font-mono text-gradient-cyan">
              {stage.time.toFixed(3)}s
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {stage.description}
            </p>
            <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(stage.time / maxTime) * 100}%`,
                  background:
                    "linear-gradient(90deg, #FF6500, #cc5100)",
                }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
