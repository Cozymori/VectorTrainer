"use client";

import { useEffect, useState, useCallback } from "react";
import {
  getFeedbackPairs,
  createFeedbackPair,
  updateFeedbackPair,
  deleteFeedbackPair,
} from "@/lib/api";
import type { FeedbackPair } from "@/lib/types";
import { MetricCard } from "@/components/metric-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function FeedbackPage() {
  const [pairs, setPairs] = useState<FeedbackPair[]>([]);
  const [loading, setLoading] = useState(true);

  // Add form
  const [newPrompt, setNewPrompt] = useState("");
  const [newBad, setNewBad] = useState("");
  const [newFixed, setNewFixed] = useState("");

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editPrompt, setEditPrompt] = useState("");
  const [editBad, setEditBad] = useState("");
  const [editFixed, setEditFixed] = useState("");

  const fetchData = useCallback(async () => {
    try {
      const data = await getFeedbackPairs();
      setPairs(data);
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

  const handleAdd = async () => {
    if (!newPrompt.trim() || !newBad.trim() || !newFixed.trim()) return;
    try {
      await createFeedbackPair({
        input_prompt: newPrompt,
        bad_output: newBad,
        fixed_output: newFixed,
      });
      setNewPrompt("");
      setNewBad("");
      setNewFixed("");
      await fetchData();
    } catch (e) {
      console.error("Failed to add feedback pair:", e);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteFeedbackPair(id);
      await fetchData();
    } catch (e) {
      console.error("Failed to delete feedback pair:", e);
    }
  };

  const startEdit = (pair: FeedbackPair) => {
    setEditingId(pair.id);
    setEditPrompt(pair.input_prompt);
    setEditBad(pair.bad_output);
    setEditFixed(pair.fixed_output);
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const handleSave = async () => {
    if (!editingId) return;
    try {
      await updateFeedbackPair(editingId, {
        input_prompt: editPrompt,
        bad_output: editBad,
        fixed_output: editFixed,
      });
      setEditingId(null);
      await fetchData();
    } catch (e) {
      console.error("Failed to update feedback pair:", e);
    }
  };

  if (loading) return <div className="text-muted-foreground">Loading...</div>;

  return (
    <div className="space-y-8">
      {/* Metric */}
      <div className="grid grid-cols-1 max-w-xs">
        <MetricCard label="Feedback Pairs" value={pairs.length} />
      </div>

      {/* Add form */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Add Feedback Pair
        </h2>
        <Card className="bg-card border-border">
          <CardContent className="p-5 space-y-4">
            <div>
              <label className="text-xs text-muted-foreground uppercase block mb-1">
                Input Prompt
              </label>
              <textarea
                className="w-full rounded-md border border-border bg-muted/30 px-3 py-2 text-sm resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-primary"
                value={newPrompt}
                onChange={(e) => setNewPrompt(e.target.value)}
                placeholder="Enter the input prompt..."
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground uppercase block mb-1">
                  Bad Output
                </label>
                <textarea
                  className="w-full rounded-md border border-border bg-red-500/5 px-3 py-2 text-sm resize-y min-h-[80px] focus:outline-none focus:ring-1 focus:ring-red-500/50"
                  value={newBad}
                  onChange={(e) => setNewBad(e.target.value)}
                  placeholder="Enter the bad output..."
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground uppercase block mb-1">
                  Fixed Output
                </label>
                <textarea
                  className="w-full rounded-md border border-border bg-green-500/5 px-3 py-2 text-sm resize-y min-h-[80px] focus:outline-none focus:ring-1 focus:ring-green-500/50"
                  value={newFixed}
                  onChange={(e) => setNewFixed(e.target.value)}
                  placeholder="Enter the fixed output..."
                />
              </div>
            </div>
            <Button onClick={handleAdd} disabled={!newPrompt.trim() || !newBad.trim() || !newFixed.trim()}>
              Add
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* Table */}
      <section>
        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4">
          Feedback Pairs
        </h2>
        {pairs.length === 0 ? (
          <div className="flex items-center justify-center h-[20vh]">
            <p className="text-muted-foreground">No feedback pairs yet.</p>
          </div>
        ) : (
          <div className="rounded-lg border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-24">ID</TableHead>
                  <TableHead>Prompt</TableHead>
                  <TableHead>Bad Output</TableHead>
                  <TableHead>Fixed Output</TableHead>
                  <TableHead className="w-32">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pairs.map((pair) =>
                  editingId === pair.id ? (
                    <TableRow key={pair.id}>
                      <TableCell className="font-mono text-sm text-primary align-top">
                        {pair.id}
                      </TableCell>
                      <TableCell className="align-top">
                        <textarea
                          className="w-full rounded-md border border-border bg-muted/30 px-2 py-1 text-sm resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-primary"
                          value={editPrompt}
                          onChange={(e) => setEditPrompt(e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="align-top">
                        <textarea
                          className="w-full rounded-md border border-border bg-red-500/5 px-2 py-1 text-sm resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-red-500/50"
                          value={editBad}
                          onChange={(e) => setEditBad(e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="align-top">
                        <textarea
                          className="w-full rounded-md border border-border bg-green-500/5 px-2 py-1 text-sm resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-green-500/50"
                          value={editFixed}
                          onChange={(e) => setEditFixed(e.target.value)}
                        />
                      </TableCell>
                      <TableCell className="align-top">
                        <div className="flex gap-1">
                          <Button size="sm" onClick={handleSave}>
                            Save
                          </Button>
                          <Button size="sm" variant="ghost" onClick={cancelEdit}>
                            Cancel
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : (
                    <TableRow key={pair.id} className="hover:bg-muted/30">
                      <TableCell className="font-mono text-sm text-primary">
                        {pair.id}
                      </TableCell>
                      <TableCell className="text-sm max-w-[200px] truncate">
                        {pair.input_prompt}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="text-red-400 border-red-500/30 font-normal max-w-[200px] truncate block"
                        >
                          {pair.bad_output}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="text-green-400 border-green-500/30 font-normal max-w-[200px] truncate block"
                        >
                          {pair.fixed_output}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => startEdit(pair)}
                          >
                            Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-red-400 hover:text-red-300"
                            onClick={() => handleDelete(pair.id)}
                          >
                            Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                )}
              </TableBody>
            </Table>
          </div>
        )}
      </section>
    </div>
  );
}
