"""Synthesizer module for VectorTrainer.

Analyzes user feedback diffs, synthesizes correction rules, and generates
physical hook script files that intercept prompts and outputs at runtime.
"""

from __future__ import annotations

import ast
import difflib
import json
import logging
import math
import os
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .types import FeedbackPair, Rule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class 1: FeedbackDiffAnalyzer
# ---------------------------------------------------------------------------


class FeedbackDiffAnalyzer:
    """Analyzes the diff between bad and fixed outputs in a FeedbackPair.

    Computes edit distance, semantic similarity (character-frequency cosine),
    and structured diff segments to quantify *what* changed.
    """

    # -- public API ----------------------------------------------------------

    def analyze(self, pair: FeedbackPair) -> Dict[str, Any]:
        """Produce a comprehensive diff analysis for a single feedback pair.

        Returns a dict containing:
            - ``edit_distance``    : Levenshtein edit distance (int)
            - ``similarity_score`` : cosine similarity of char-frequency vectors (float)
            - ``diff_summary``     : human-readable summary of what changed (str)
            - ``diff_segments``    : list of added / removed / changed segments
            - ``input_prompt``     : the original prompt (pass-through for downstream)
            - ``context``          : original context dict (pass-through)
        """
        bad = pair.bad_output
        fixed = pair.fixed_output

        edit_dist = self._levenshtein_distance(bad, fixed)
        similarity = self._char_frequency_cosine(bad, fixed)
        segments = self._extract_diff_segments(bad, fixed)
        summary = self._build_diff_summary(segments, edit_dist)

        return {
            "edit_distance": edit_dist,
            "similarity_score": similarity,
            "diff_summary": summary,
            "diff_segments": segments,
            "input_prompt": pair.input_prompt,
            "context": pair.context,
            "bad_output": bad,
            "fixed_output": fixed,
        }

    # -- Levenshtein ---------------------------------------------------------

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Standard dynamic-programming Levenshtein edit distance.

        Time complexity: O(m*n), space complexity: O(min(m, n)) via the
        two-row optimisation.
        """
        if s1 == s2:
            return 0
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Keep the shorter string in the inner loop for space efficiency.
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        prev_row: List[int] = list(range(len1 + 1))
        for j in range(1, len2 + 1):
            curr_row: List[int] = [j] + [0] * len1
            for i in range(1, len1 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[i] = min(
                    curr_row[i - 1] + 1,       # insertion
                    prev_row[i] + 1,            # deletion
                    prev_row[i - 1] + cost,     # substitution
                )
            prev_row = curr_row

        return prev_row[len1]

    # -- Diff segments -------------------------------------------------------

    @staticmethod
    def _extract_diff_segments(bad: str, fixed: str) -> List[Dict[str, Any]]:
        """Identify added / removed / changed segments using *difflib*.

        Returns a list of dicts, each with:
            - ``type``  : ``"added"`` | ``"removed"`` | ``"changed"``
            - ``bad``   : text from the bad output (empty for additions)
            - ``fixed`` : text from the fixed output (empty for removals)
        """
        segments: List[Dict[str, Any]] = []
        matcher = difflib.SequenceMatcher(None, bad, fixed)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            elif tag == "insert":
                segments.append({
                    "type": "added",
                    "bad": "",
                    "fixed": fixed[j1:j2],
                })
            elif tag == "delete":
                segments.append({
                    "type": "removed",
                    "bad": bad[i1:i2],
                    "fixed": "",
                })
            elif tag == "replace":
                segments.append({
                    "type": "changed",
                    "bad": bad[i1:i2],
                    "fixed": fixed[j1:j2],
                })

        return segments

    # -- Semantic similarity placeholder (char-frequency cosine) -------------

    @staticmethod
    def _char_frequency_cosine(s1: str, s2: str) -> float:
        """Cosine similarity of character-frequency vectors.

        This is a *placeholder* for a real embedding-based similarity score.
        Returns a float in [0.0, 1.0].
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Build frequency maps.
        chars = set(s1) | set(s2)
        freq1 = {ch: s1.count(ch) for ch in chars}
        freq2 = {ch: s2.count(ch) for ch in chars}

        dot = sum(freq1.get(ch, 0) * freq2.get(ch, 0) for ch in chars)
        mag1 = math.sqrt(sum(v * v for v in freq1.values()))
        mag2 = math.sqrt(sum(v * v for v in freq2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    # -- Summary builder -----------------------------------------------------

    @staticmethod
    def _build_diff_summary(segments: List[Dict[str, Any]], edit_distance: int) -> str:
        """Create a short human-readable summary of what changed."""
        added = sum(1 for s in segments if s["type"] == "added")
        removed = sum(1 for s in segments if s["type"] == "removed")
        changed = sum(1 for s in segments if s["type"] == "changed")

        parts: List[str] = []
        if added:
            parts.append(f"{added} addition(s)")
        if removed:
            parts.append(f"{removed} removal(s)")
        if changed:
            parts.append(f"{changed} replacement(s)")

        detail = ", ".join(parts) if parts else "no changes"
        return f"Edit distance {edit_distance}: {detail}"


# ---------------------------------------------------------------------------
# Class 2: RuleSetSynthesizer
# ---------------------------------------------------------------------------


class RuleSetSynthesizer:
    """Converts analysed diffs into a coherent set of :class:`Rule` objects.

    Supports both heuristic-based synthesis and optional LLM-based synthesis
    via an OpenAI-compatible client.
    """

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.llm_client = llm_client
        self.model = model

    # -- Heuristic synthesis -------------------------------------------------

    def synthesize(self, diffs: List[Dict[str, Any]]) -> List[Rule]:
        """Convert analysed diffs into :class:`Rule` objects.

        Each diff produces one rule whose *condition* describes when the rule
        applies and whose *action* describes the transformation.
        """
        rules: List[Rule] = []

        for idx, diff in enumerate(diffs):
            condition = self._build_condition(diff)
            action = self._build_action(diff)
            description = self._build_description(diff)

            rule = Rule(
                rule_id=f"R-{idx:03d}",
                description=description,
                condition=condition,
                action=action,
                priority=self._compute_priority(diff),
                source_pair_index=idx,
            )
            rules.append(rule)

        # Detect and resolve contradictions.
        contradictions = self._detect_contradictions(rules)
        if contradictions:
            logger.info(
                "Detected %d contradiction(s) among %d rules; merging.",
                len(contradictions),
                len(rules),
            )
            rules = self._merge_contradicting_rules(rules, contradictions)

        return rules

    # -- LLM-based synthesis -------------------------------------------------

    def synthesize_with_llm(self, diffs: List[Dict[str, Any]]) -> List[Rule]:
        """Use an OpenAI-compatible API to generate rules from diffs.

        Falls back to the heuristic ``synthesize()`` method if the LLM call
        fails for any reason (network error, parse failure, etc.).
        """
        if self.llm_client is None:
            logger.warning("No LLM client provided; falling back to heuristic synthesis.")
            return self.synthesize(diffs)

        prompt = self._build_llm_prompt(diffs)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at analysing text diffs and producing "
                            "structured correction rules. Respond ONLY with a JSON array "
                            "of rule objects. Each object must have the keys: "
                            "rule_id, description, condition, action, priority."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw_text = response.choices[0].message.content.strip()
            rules = self._parse_llm_rules(raw_text, diffs)
            logger.info("LLM synthesis produced %d rule(s).", len(rules))
            return rules
        except Exception:
            logger.exception("LLM synthesis failed; falling back to heuristic synthesis.")
            return self.synthesize(diffs)

    # -- Contradiction detection ---------------------------------------------

    @staticmethod
    def _detect_contradictions(rules: List[Rule]) -> List[tuple[int, int]]:
        """Find pairs of rules that appear to contradict each other.

        Heuristic: two rules contradict when their conditions are very similar
        (sequence ratio > 0.8) but their actions are semantically opposed
        (sequence ratio < 0.3).
        """
        contradictions: List[tuple[int, int]] = []

        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                cond_ratio = difflib.SequenceMatcher(
                    None, rules[i].condition, rules[j].condition
                ).ratio()
                action_ratio = difflib.SequenceMatcher(
                    None, rules[i].action, rules[j].action
                ).ratio()

                if cond_ratio > 0.8 and action_ratio < 0.3:
                    contradictions.append((i, j))

        return contradictions

    # -- Contradiction merging -----------------------------------------------

    @staticmethod
    def _merge_contradicting_rules(
        rules: List[Rule],
        contradictions: List[tuple[int, int]],
    ) -> List[Rule]:
        """Merge contradicting rule pairs.

        Strategy: keep the rule with the higher priority.  If priorities are
        equal, merge descriptions and keep the first rule's action.  Indices
        marked for removal are dropped from the final list.
        """
        indices_to_remove: set[int] = set()

        for i, j in contradictions:
            if i in indices_to_remove or j in indices_to_remove:
                continue

            rule_a, rule_b = rules[i], rules[j]

            if rule_b.priority > rule_a.priority:
                # Keep rule_b, mark rule_a for removal.
                rule_b.description = (
                    f"{rule_b.description} [merged with {rule_a.rule_id}]"
                )
                indices_to_remove.add(i)
            else:
                # Keep rule_a (equal priority defaults here too).
                rule_a.description = (
                    f"{rule_a.description} [merged with {rule_b.rule_id}]"
                )
                indices_to_remove.add(j)

        return [r for idx, r in enumerate(rules) if idx not in indices_to_remove]

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _build_condition(diff: Dict[str, Any]) -> str:
        """Derive a rule condition from an analysed diff."""
        segments = diff.get("diff_segments", [])
        prompt = diff.get("input_prompt", "")

        if not segments:
            return "always"

        # Build condition from the first removed/changed segment and the prompt.
        for seg in segments:
            if seg["type"] in ("removed", "changed"):
                bad_snippet = seg["bad"][:120]
                if prompt:
                    return (
                        f"When the output contains \"{bad_snippet}\" "
                        f"in response to prompt resembling \"{prompt[:80]}\""
                    )
                return f"When the output contains \"{bad_snippet}\""

        return f"When responding to prompt resembling \"{prompt[:80]}\"" if prompt else "always"

    @staticmethod
    def _build_action(diff: Dict[str, Any]) -> str:
        """Derive a rule action from an analysed diff."""
        segments = diff.get("diff_segments", [])

        actions: List[str] = []
        for seg in segments:
            if seg["type"] == "added":
                actions.append(f"Add: \"{seg['fixed'][:100]}\"")
            elif seg["type"] == "removed":
                actions.append(f"Remove: \"{seg['bad'][:100]}\"")
            elif seg["type"] == "changed":
                actions.append(
                    f"Replace \"{seg['bad'][:60]}\" with \"{seg['fixed'][:60]}\""
                )

        return "; ".join(actions) if actions else "No action required"

    @staticmethod
    def _build_description(diff: Dict[str, Any]) -> str:
        """Create a human-readable description for a rule."""
        summary = diff.get("diff_summary", "")
        prompt_snippet = diff.get("input_prompt", "")[:60]
        if prompt_snippet:
            return f"Correction rule for prompt \"{prompt_snippet}\": {summary}"
        return f"Correction rule: {summary}"

    @staticmethod
    def _compute_priority(diff: Dict[str, Any]) -> int:
        """Assign priority based on edit distance (larger change = higher priority)."""
        dist = diff.get("edit_distance", 0)
        if dist > 100:
            return 3
        if dist > 30:
            return 2
        if dist > 0:
            return 1
        return 0

    @staticmethod
    def _build_llm_prompt(diffs: List[Dict[str, Any]]) -> str:
        """Build the user-message prompt sent to the LLM."""
        entries: List[str] = []
        for idx, diff in enumerate(diffs):
            entry = (
                f"Diff #{idx}:\n"
                f"  Input prompt : {diff.get('input_prompt', 'N/A')}\n"
                f"  Bad output   : {diff.get('bad_output', 'N/A')[:300]}\n"
                f"  Fixed output : {diff.get('fixed_output', 'N/A')[:300]}\n"
                f"  Edit distance: {diff.get('edit_distance', 'N/A')}\n"
                f"  Summary      : {diff.get('diff_summary', 'N/A')}\n"
            )
            entries.append(entry)

        return (
            "Analyse the following feedback diffs and produce a JSON array of "
            "correction rules.\n\n" + "\n".join(entries)
        )

    @staticmethod
    def _parse_llm_rules(raw_text: str, diffs: List[Dict[str, Any]]) -> List[Rule]:
        """Parse the LLM's JSON response into :class:`Rule` objects."""
        # Strip markdown code fences if present.
        text = raw_text.strip()
        if text.startswith("```"):
            # Remove opening fence (possibly with language tag).
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("LLM response is not a JSON array.")

        rules: List[Rule] = []
        for idx, item in enumerate(items):
            rule = Rule(
                rule_id=item.get("rule_id", f"R-{idx:03d}"),
                description=item.get("description", ""),
                condition=item.get("condition", "always"),
                action=item.get("action", ""),
                priority=int(item.get("priority", 0)),
                source_pair_index=idx if idx < len(diffs) else -1,
            )
            rules.append(rule)

        return rules


# ---------------------------------------------------------------------------
# Class 3: HookScriptGenerator
# ---------------------------------------------------------------------------


class HookScriptGenerator:
    """Generates a self-contained Python hook script from a list of rules.

    The generated file exposes two public functions that a runtime can call
    to intercept prompts and model outputs.
    """

    SCRIPT_FILENAME = "generated_prompt_hook.py"

    # -- public API ----------------------------------------------------------

    def generate(self, rules: List[Rule], output_dir: str) -> str:
        """Generate the hook script file and return its absolute path.

        Raises
        ------
        SyntaxError
            If the generated source fails ``ast.parse()`` validation.
        OSError
            If the output directory cannot be created or written to.
        """
        os.makedirs(output_dir, exist_ok=True)

        source = self._build_script_source(rules)

        if not self._validate_syntax(source):
            raise SyntaxError(
                "Generated hook script contains invalid Python syntax."
            )

        file_path = os.path.join(output_dir, self.SCRIPT_FILENAME)
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(source)

        logger.info("Hook script written to %s (%d rules)", file_path, len(rules))
        return file_path

    # -- Script builder ------------------------------------------------------

    def _build_script_source(self, rules: List[Rule]) -> str:
        """Build the complete Python source code as a string."""
        timestamp = datetime.now(timezone.utc).isoformat()

        rules_literal = self._rules_to_literal(rules)

        source = textwrap.dedent(f'''\
            """Auto-generated prompt hook script.

            Generated by VectorTrainer HookScriptGenerator
            Timestamp: {timestamp}
            Total rules: {len(rules)}

            Do NOT edit manually — this file is regenerated on every synthesis run.
            """

            from __future__ import annotations

            import re
            from typing import Any, Dict, List


            # ---------------------------------------------------------------------------
            # Rules
            # ---------------------------------------------------------------------------

            RULES: List[Dict[str, Any]] = {rules_literal}


            # ---------------------------------------------------------------------------
            # Public intercept functions
            # ---------------------------------------------------------------------------


            def intercept_prompt(user_input: str, context: dict) -> str:
                """Pre-process the user prompt before it reaches the model.

                Applies all matching rules (sorted by priority descending) to the
                user input and returns the modified prompt.
                """
                sorted_rules = sorted(RULES, key=lambda r: r["priority"], reverse=True)
                modified = user_input

                for rule in sorted_rules:
                    condition = rule.get("condition", "")
                    action = rule.get("action", "")

                    if _condition_matches(condition, modified, context):
                        modified = _apply_action(action, modified)

                return modified


            def intercept_output(model_output: str, context: dict) -> str:
                """Post-process the model output before returning it to the user.

                Applies all matching rules (sorted by priority descending) to the
                model output and returns the corrected text.
                """
                sorted_rules = sorted(RULES, key=lambda r: r["priority"], reverse=True)
                modified = model_output

                for rule in sorted_rules:
                    condition = rule.get("condition", "")
                    action = rule.get("action", "")

                    if _condition_matches(condition, modified, context):
                        modified = _apply_action(action, modified)

                return modified


            # ---------------------------------------------------------------------------
            # Internal helpers
            # ---------------------------------------------------------------------------


            def _condition_matches(condition: str, text: str, context: dict) -> bool:
                """Check whether a rule condition matches the given text and context."""
                if not condition or condition.lower() == "always":
                    return True

                # Simple substring match: look for quoted substrings in the condition.
                quoted = re.findall(r'"([^"]*)"', condition)
                for snippet in quoted:
                    if snippet and snippet.lower() in text.lower():
                        return True

                return False


            def _apply_action(action: str, text: str) -> str:
                """Apply a rule action to the given text.

                Supports the following action prefixes:
                  - ``Replace "X" with "Y"``
                  - ``Remove: "X"``
                  - ``Add: "X"``
                """
                parts = [a.strip() for a in action.split(";")]
                modified = text

                for part in parts:
                    if part.lower().startswith("replace"):
                        match = re.search(r'"([^"]*)"\\s+with\\s+"([^"]*)"', part)
                        if match:
                            old, new = match.group(1), match.group(2)
                            modified = modified.replace(old, new)
                    elif part.lower().startswith("remove"):
                        match = re.search(r'"([^"]*)"', part)
                        if match:
                            modified = modified.replace(match.group(1), "")
                    elif part.lower().startswith("add"):
                        match = re.search(r'"([^"]*)"', part)
                        if match:
                            modified = modified + " " + match.group(1)

                return modified
        ''')

        return source

    # -- Syntax validation ---------------------------------------------------

    @staticmethod
    def _validate_syntax(source: str) -> bool:
        """Return ``True`` if *source* is syntactically valid Python."""
        try:
            ast.parse(source)
            return True
        except SyntaxError as exc:
            logger.error("Syntax validation failed: %s", exc)
            return False

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _rules_to_literal(rules: List[Rule]) -> str:
        """Serialize a list of :class:`Rule` objects to a Python list literal."""
        items: List[Dict[str, Any]] = []
        for rule in rules:
            items.append({
                "rule_id": rule.rule_id,
                "description": rule.description,
                "condition": rule.condition,
                "action": rule.action,
                "priority": rule.priority,
                "source_pair_index": rule.source_pair_index,
            })
        return repr(items)


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------


def run_synthesis_pipeline(
    pairs: List[FeedbackPair],
    output_dir: str,
    llm_client: Any = None,
    use_llm: bool = False,
) -> str:
    """Orchestrate the full synthesis pipeline.

    1. Analyse all feedback pairs to produce diffs.
    2. Synthesize diffs into rules (heuristic or LLM-based).
    3. Generate the hook script and write it to *output_dir*.

    Parameters
    ----------
    pairs:
        List of :class:`FeedbackPair` objects to process.
    output_dir:
        Directory where the hook script will be written.
    llm_client:
        Optional OpenAI-compatible client for LLM-based synthesis.
    use_llm:
        If ``True`` and *llm_client* is provided, use LLM-based synthesis.

    Returns
    -------
    str
        Absolute path to the generated hook script.
    """
    logger.info("Starting synthesis pipeline with %d feedback pair(s).", len(pairs))

    # Step 1: Analyse.
    analyzer = FeedbackDiffAnalyzer()
    diffs: List[Dict[str, Any]] = [analyzer.analyze(pair) for pair in pairs]
    logger.info("Analysis complete: %d diff(s) produced.", len(diffs))

    # Step 2: Synthesize.
    synthesizer = RuleSetSynthesizer(llm_client=llm_client)
    if use_llm and llm_client is not None:
        rules = synthesizer.synthesize_with_llm(diffs)
    else:
        rules = synthesizer.synthesize(diffs)
    logger.info("Synthesis complete: %d rule(s) produced.", len(rules))

    # Step 3: Generate hook script.
    generator = HookScriptGenerator()
    script_path = generator.generate(rules, output_dir)
    logger.info("Pipeline finished. Hook script: %s", script_path)

    return script_path
