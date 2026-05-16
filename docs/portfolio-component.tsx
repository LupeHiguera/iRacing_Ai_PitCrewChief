/**
 * iRacing AI Pit Crew Chief — Eval Comparison Dashboard
 *
 * Drop-in React/Next.js component for higuera.io/projects/iracing-strategist.
 *
 * Two ways to use it:
 *
 * 1) Import the JSON directly (recommended for static sites):
 *    - Copy `eval_comprehensive.json` into your site's `public/` or alongside this file
 *    - import data from "./eval_comprehensive.json"
 *    - <EvalDashboard data={data} />
 *
 * 2) Fetch at runtime:
 *    - Host the JSON anywhere (GitHub Pages, raw.githubusercontent.com)
 *    - <EvalDashboard src="https://lupehiguera.github.io/iRacing_Ai_PitCrewChief/eval_comprehensive.json" />
 *
 * Styling: uses inline CSS (CSS-in-JS-style) so it works with any framework.
 * If you use Tailwind / CSS modules / styled-components, swap the `style` props.
 *
 * Tested with: Next.js 14 App Router, React 18+, TypeScript 5+.
 */

"use client";

import { useEffect, useMemo, useState } from "react";

// ---------- Types ----------
type Metrics = {
  word_count: number;
  is_concise: boolean;
  is_tts_suitable: boolean;
  has_telemetry_reference: boolean;
  has_track_reference: boolean;
  has_car_appropriate_advice: boolean;
  urgency_appropriate: boolean;
  contains_required_element: boolean;
  expected_elements_found: number;
  expected_elements_total: number;
  contains_hallucination: boolean;
  is_actionable: boolean;
  is_specific: boolean;
  references_correct_values: boolean;
  composite_score: number;
};

type Case = {
  name: string;
  category: string;
  urgency: string;
  input_summary: string;
  base_response: string;
  finetuned_response: string;
  base_metrics: Metrics;
  finetuned_metrics: Metrics;
  base_latency_ms: number;
  finetuned_latency_ms: number;
};

type EvalData = {
  summary: {
    total_cases: number;
    base_avg_score: number;
    ft_avg_score: number;
    ft_wins: number;
    base_wins: number;
  };
  results: Case[];
};

type Props = { data?: EvalData; src?: string };

// ---------- Constants ----------
const CATEGORY_LABELS: Record<string, string> = {
  fuel_critical: "Fuel Critical",
  fuel_warning: "Fuel Warning",
  tire_critical: "Tire Critical",
  tire_warning: "Tire Warning",
  tire_cold: "Tire Cold",
  position_battle: "Position Battle",
  gap_management: "Gap Management",
  pit_approach: "Pit Approach",
  pace_feedback: "Pace Feedback",
  routine: "Routine",
};

const METRIC_ROWS: Array<{ key: keyof Metrics; label: string; kind: "num" | "bool" | "bool-inverse" }> = [
  { key: "composite_score", label: "Composite score", kind: "num" },
  { key: "word_count", label: "Word count", kind: "num" },
  { key: "is_concise", label: "Concise (< 40 words)", kind: "bool" },
  { key: "is_tts_suitable", label: "TTS suitable", kind: "bool" },
  { key: "has_telemetry_reference", label: "References telemetry", kind: "bool" },
  { key: "has_track_reference", label: "References track corner", kind: "bool" },
  { key: "has_car_appropriate_advice", label: "Car-appropriate advice", kind: "bool" },
  { key: "urgency_appropriate", label: "Urgency matches", kind: "bool" },
  { key: "contains_required_element", label: "Has required keyword", kind: "bool" },
  { key: "is_actionable", label: "Actionable", kind: "bool" },
  { key: "is_specific", label: "Specific (cites numbers)", kind: "bool" },
  { key: "references_correct_values", label: "Numbers match input", kind: "bool" },
  { key: "contains_hallucination", label: "Hallucinations", kind: "bool-inverse" },
];

// ---------- Theme tokens ----------
// Override these via CSS variables on a parent element to match the site palette:
//   --eval-bg, --eval-bg-elev, --eval-bg-2, --eval-border,
//   --eval-text, --eval-text-dim, --eval-text-mute,
//   --eval-accent, --eval-good, --eval-bad
const VARS = {
  "--eval-bg": "#0b0d10",
  "--eval-bg-elev": "#14181d",
  "--eval-bg-2": "#1b2128",
  "--eval-border": "#262d36",
  "--eval-text": "#e6e9ed",
  "--eval-text-dim": "#9aa3ad",
  "--eval-text-mute": "#6a727c",
  "--eval-accent": "#60a5fa",
  "--eval-good": "#4ade80",
  "--eval-bad": "#f87171",
} as const;

// ---------- Component ----------
export default function EvalDashboard({ data: dataProp, src }: Props) {
  const [data, setData] = useState<EvalData | null>(dataProp ?? null);
  const [err, setErr] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>("all");
  const [activeIdx, setActiveIdx] = useState<number>(0);

  useEffect(() => {
    if (dataProp || !src) return;
    let cancelled = false;
    fetch(src)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((json: EvalData) => !cancelled && setData(json))
      .catch((e) => !cancelled && setErr(e.message));
    return () => { cancelled = true; };
  }, [dataProp, src]);

  const filtered = useMemo(() => {
    if (!data) return [];
    if (activeFilter === "all") return data.results.map((c, i) => ({ c, i }));
    return data.results.map((c, i) => ({ c, i })).filter((o) => o.c.category === activeFilter);
  }, [data, activeFilter]);

  useEffect(() => {
    if (filtered.length > 0 && !filtered.some((o) => o.i === activeIdx)) {
      setActiveIdx(filtered[0].i);
    }
  }, [filtered, activeIdx]);

  if (err) return <div style={{ ...rootStyle, padding: 40, color: "var(--eval-bad)" }}>Failed to load eval data: {err}</div>;
  if (!data) return <div style={{ ...rootStyle, padding: 40, color: "var(--eval-text-mute)" }}>Loading eval data…</div>;

  const active = data.results[activeIdx];
  const baseLat = avgLatency(data, "base_latency_ms");
  const ftLat = avgLatency(data, "finetuned_latency_ms");
  const halluc = countHalluc(data);

  return (
    <div style={rootStyle as React.CSSProperties}>
      {/* Summary cards */}
      <div style={summaryGrid}>
        <Card label="Composite Score">
          <span style={{ color: "var(--eval-bad)" }}>{data.summary.base_avg_score.toFixed(1)}</span>
          <Arrow />
          <span style={{ color: "var(--eval-good)" }}>{data.summary.ft_avg_score.toFixed(1)}</span>
          <Sub>out of 100 · +{(data.summary.ft_avg_score - data.summary.base_avg_score).toFixed(1)} avg lift</Sub>
        </Card>
        <Card label="Fine-tuned Wins">
          <span style={{ color: "var(--eval-good)" }}>{data.summary.ft_wins} / {data.summary.total_cases}</span>
          <Sub>{((data.summary.ft_wins / data.summary.total_cases) * 100).toFixed(1)}% across all categories</Sub>
        </Card>
        <Card label="Hallucinations">
          <span style={{ color: "var(--eval-bad)" }}>{halluc.base}</span>
          <Arrow />
          <span style={{ color: "var(--eval-good)" }}>{halluc.ft}</span>
          <Sub>out of {data.summary.total_cases} cases</Sub>
        </Card>
        <Card label="Avg Latency">
          <span style={{ color: "var(--eval-text-dim)" }}>{(baseLat / 1000).toFixed(2)}s</span>
          <Arrow />
          <span style={{ color: "var(--eval-good)" }}>{(ftLat / 1000).toFixed(2)}s</span>
          <Sub>model output time</Sub>
        </Card>
      </div>

      {/* Filters */}
      <div style={filterRow}>
        {(["all", ...Object.keys(CATEGORY_LABELS)] as string[]).map((c) => {
          const count = c === "all" ? data.results.length : data.results.filter((r) => r.category === c).length;
          const label = c === "all" ? "All categories" : CATEGORY_LABELS[c];
          const isActive = c === activeFilter;
          return (
            <button
              key={c}
              onClick={() => setActiveFilter(c)}
              style={{
                ...chipStyle,
                background: isActive ? "var(--eval-accent)" : "var(--eval-bg-elev)",
                color: isActive ? "#0b0d10" : "var(--eval-text-dim)",
                borderColor: isActive ? "var(--eval-accent)" : "var(--eval-border)",
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {label} <span style={{ opacity: 0.7 }}>({count})</span>
            </button>
          );
        })}
      </div>

      {/* Grid */}
      <div style={gridStyle}>
        {/* Sidebar */}
        <aside style={sidebarStyle}>
          {filtered.map(({ c, i }) => {
            const delta = c.finetuned_metrics.composite_score - c.base_metrics.composite_score;
            const isActive = i === activeIdx;
            return (
              <div
                key={i}
                onClick={() => setActiveIdx(i)}
                style={{
                  ...caseItemStyle,
                  background: isActive ? "var(--eval-bg-2)" : "transparent",
                  borderColor: isActive ? "var(--eval-accent)" : "transparent",
                }}
              >
                <div style={{ fontSize: 13.5, fontWeight: 500, color: "var(--eval-text)", marginBottom: 2 }}>{c.name}</div>
                <div style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 11, color: "var(--eval-text-mute)" }}>
                  <span style={catTagStyle}>{CATEGORY_LABELS[c.category] || c.category}</span>
                  <span style={{ textTransform: "uppercase", fontSize: 10, letterSpacing: 0.5 }}>{c.urgency}</span>
                  <span style={{ marginLeft: "auto", color: "var(--eval-good)", fontWeight: 600 }}>
                    {delta >= 0 ? "+" : ""}{delta.toFixed(0)}
                  </span>
                </div>
              </div>
            );
          })}
        </aside>

        {/* Detail */}
        <main style={detailStyle}>
          <h3 style={{ margin: 0, fontSize: 20 }}>{active.name}</h3>
          <div style={{ color: "var(--eval-text-dim)", marginBottom: 20, fontSize: 14 }}>{active.input_summary}</div>

          <div style={responsesGrid}>
            <ResponseCard kind="base" score={active.base_metrics.composite_score} text={active.base_response} latency={active.base_latency_ms} />
            <ResponseCard kind="ft" score={active.finetuned_metrics.composite_score} text={active.finetuned_response} latency={active.finetuned_latency_ms} />
          </div>

          <h4 style={{ margin: "0 0 12px", fontSize: 12, textTransform: "uppercase", letterSpacing: 1, color: "var(--eval-text-mute)" }}>
            Metric breakdown
          </h4>
          {METRIC_ROWS.map((m) => (
            <div key={m.key} style={metricRowStyle}>
              <div style={{ color: "var(--eval-text-dim)" }}>{m.label}</div>
              <div>{renderBadge(active.base_metrics[m.key], m.kind)}</div>
              <div>{renderBadge(active.finetuned_metrics[m.key], m.kind)}</div>
            </div>
          ))}
        </main>
      </div>
    </div>
  );
}

// ---------- Sub-components ----------
function Card({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={cardStyle}>
      <div style={{ color: "var(--eval-text-mute)", fontSize: 11, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, lineHeight: 1 }}>{children}</div>
    </div>
  );
}

function Sub({ children }: { children: React.ReactNode }) {
  return <div style={{ color: "var(--eval-text-dim)", fontSize: 13, marginTop: 4, fontWeight: 400 }}>{children}</div>;
}

function Arrow() {
  return <span style={{ color: "var(--eval-text-mute)", fontWeight: 400, fontSize: 22, margin: "0 4px" }}>→</span>;
}

function ResponseCard({ kind, score, text, latency }: { kind: "base" | "ft"; score: number; text: string; latency: number }) {
  const isBase = kind === "base";
  return (
    <div style={{
      ...responseCardStyle,
      borderLeft: `3px solid ${isBase ? "var(--eval-bad)" : "var(--eval-good)"}`,
    }}>
      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, color: "var(--eval-text-mute)", marginBottom: 6 }}>
        {isBase ? "Base Llama 3.1 8B" : "Fine-tuned (QLoRA r=16)"}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 10, color: isBase ? "var(--eval-bad)" : "var(--eval-good)" }}>
        {score.toFixed(0)} <span style={{ fontSize: 13, color: "var(--eval-text-mute)", fontWeight: 400 }}>/ 100</span>
      </div>
      <div style={{ fontSize: 14, lineHeight: 1.55, whiteSpace: "pre-wrap", color: "var(--eval-text)" }}>{text}</div>
      {latency > 0 && (
        <div style={{ marginTop: 12, fontSize: 12, color: "var(--eval-text-mute)" }}>{(latency / 1000).toFixed(2)}s latency</div>
      )}
    </div>
  );
}

function renderBadge(v: number | boolean, kind: "num" | "bool" | "bool-inverse") {
  if (kind === "num") {
    const n = typeof v === "number" ? v : 0;
    return <span style={badgeStyle("num")}>{Number.isInteger(n) ? n : n.toFixed(1)}</span>;
  }
  if (kind === "bool") return v ? <span style={badgeStyle("yes")}>yes</span> : <span style={badgeStyle("no")}>no</span>;
  // bool-inverse: contains_hallucination → "yes" is bad
  return v ? <span style={badgeStyle("no")}>yes</span> : <span style={badgeStyle("yes")}>clean</span>;
}

function badgeStyle(kind: "yes" | "no" | "num"): React.CSSProperties {
  const bg = kind === "yes" ? "rgba(74, 222, 128, 0.15)" : kind === "no" ? "rgba(248, 113, 113, 0.15)" : "var(--eval-bg)";
  const color = kind === "yes" ? "var(--eval-good)" : kind === "no" ? "var(--eval-bad)" : "var(--eval-text)";
  return { display: "inline-block", minWidth: 24, padding: "2px 8px", borderRadius: 4, fontSize: 12, textAlign: "center", fontWeight: 600, background: bg, color };
}

// ---------- Helpers ----------
function avgLatency(d: EvalData, field: "base_latency_ms" | "finetuned_latency_ms"): number {
  const vals = d.results.map((r) => r[field] || 0).filter((v) => v > 0);
  return vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
}

function countHalluc(d: EvalData): { base: number; ft: number } {
  let base = 0, ft = 0;
  for (const r of d.results) {
    if (r.base_metrics.contains_hallucination) base++;
    if (r.finetuned_metrics.contains_hallucination) ft++;
  }
  return { base, ft };
}

// ---------- Styles ----------
const rootStyle: React.CSSProperties = {
  ...VARS,
  background: "var(--eval-bg)",
  color: "var(--eval-text)",
  padding: 24,
  borderRadius: 12,
  border: "1px solid var(--eval-border)",
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  fontSize: 15,
  lineHeight: 1.5,
};

const summaryGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
  gap: 12,
  marginBottom: 24,
};

const cardStyle: React.CSSProperties = {
  background: "var(--eval-bg-elev)",
  border: "1px solid var(--eval-border)",
  borderRadius: 10,
  padding: 16,
};

const filterRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  marginBottom: 16,
};

const chipStyle: React.CSSProperties = {
  border: "1px solid",
  padding: "6px 12px",
  borderRadius: 999,
  cursor: "pointer",
  fontSize: 13,
  transition: "all 0.15s",
};

const gridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(260px, 320px) 1fr",
  gap: 16,
};

const sidebarStyle: React.CSSProperties = {
  background: "var(--eval-bg-elev)",
  border: "1px solid var(--eval-border)",
  borderRadius: 10,
  padding: 8,
  maxHeight: "70vh",
  overflowY: "auto",
};

const caseItemStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderRadius: 6,
  cursor: "pointer",
  border: "1px solid transparent",
  marginBottom: 2,
};

const catTagStyle: React.CSSProperties = {
  padding: "1px 6px",
  borderRadius: 4,
  background: "var(--eval-bg)",
  color: "var(--eval-text-dim)",
};

const detailStyle: React.CSSProperties = {
  background: "var(--eval-bg-elev)",
  border: "1px solid var(--eval-border)",
  borderRadius: 10,
  padding: 24,
  minWidth: 0,
};

const responsesGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 16,
  marginBottom: 24,
};

const responseCardStyle: React.CSSProperties = {
  background: "var(--eval-bg-2)",
  border: "1px solid var(--eval-border)",
  borderRadius: 8,
  padding: 16,
};

const metricRowStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(160px, 200px) 1fr 1fr",
  gap: 12,
  alignItems: "center",
  padding: "6px 0",
  borderBottom: "1px solid var(--eval-border)",
  fontSize: 13,
};
