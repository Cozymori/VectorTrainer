"use client";

export function Header() {
  return (
    <header className="h-14 border-b border-border flex items-center px-6">
      <h1 className="text-lg font-semibold tracking-tight">
        <span className="text-gradient-cyan">VectorTrainer</span>
      </h1>
      <span className="ml-3 text-xs text-muted-foreground">
        Self-Evolving AI Loop Dashboard
      </span>
    </header>
  );
}
