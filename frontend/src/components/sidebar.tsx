"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { useState } from "react";
import { runPipeline } from "@/lib/api";

const NAV_ITEMS = [
  { href: "/", label: "Pipeline" },
  { href: "/golden-data", label: "Golden Data" },
  { href: "/rules", label: "Rules" },
  { href: "/hook-script", label: "Hook Script" },
  { href: "/fine-tuning", label: "Fine-tuning" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [epsilon, setEpsilon] = useState(0.3);
  const [topK, setTopK] = useState(50);
  const [running, setRunning] = useState(false);

  const handleRun = async () => {
    setRunning(true);
    try {
      await runPipeline({
        epsilon,
        top_k: topK,
        use_real: false,
        function_name: "generate_review_summary",
      });
      window.dispatchEvent(new Event("pipeline-updated"));
    } catch (e) {
      console.error("Pipeline run failed:", e);
    } finally {
      setRunning(false);
    }
  };

  return (
    <aside className="w-60 h-screen fixed left-0 top-0 border-r flex flex-col"
      style={{
        backgroundColor: "#0B192C",
        borderColor: "rgba(30,62,98,0.4)",
      }}
    >
      <div className="p-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold"
            style={{
              background: "linear-gradient(135deg, #FF6500, #cc5100)",
              color: "#000000",
            }}
          >
            VT
          </div>
          <span className="font-semibold text-sm tracking-tight">VectorTrainer</span>
        </Link>
      </div>

      <Separator className="opacity-30" />

      <nav className="flex-1 p-2 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`block px-3 py-2 rounded-md text-sm transition-colors ${
                active
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <Separator className="opacity-30" />

      <div className="p-4 space-y-4">
        <div>
          <label className="text-xs text-muted-foreground block mb-2">
            Epsilon: {epsilon.toFixed(2)}
          </label>
          <Slider
            value={[epsilon]}
            onValueChange={([v]) => setEpsilon(v)}
            min={0.1}
            max={0.5}
            step={0.01}
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground block mb-2">
            Top K: {topK}
          </label>
          <Slider
            value={[topK]}
            onValueChange={([v]) => setTopK(v)}
            min={10}
            max={100}
            step={5}
          />
        </div>
        <Button
          className="w-full"
          onClick={handleRun}
          disabled={running}
        >
          {running ? "Running..." : "Run Pipeline"}
        </Button>
      </div>
    </aside>
  );
}
