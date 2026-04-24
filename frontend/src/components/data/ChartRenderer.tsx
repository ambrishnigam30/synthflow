"use client";

import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ── AI feature colors ─────────────────────────────────────────────────────────

const AI_COLORS = [
  "#f54e00", // orange (primary)
  "#dfa88f", // thinking peach
  "#9fc9a2", // search sage
  "#9fbbe0", // read blue
  "#c0a8dd", // edit lavender
  "#c08532", // gold
];

// ── Types ─────────────────────────────────────────────────────────────────────

type ChartType = "bar" | "line" | "scatter" | "auto";

interface ChartRendererProps {
  data: Record<string, unknown>[];
  xKey?: string;
  yKeys?: string[];
  chartType?: ChartType;
  height?: number;
  title?: string;
}

// ── Auto-detect chart type ────────────────────────────────────────────────────

function detectChartType(
  data: Record<string, unknown>[],
  xKey: string,
  yKeys: string[]
): "bar" | "line" | "scatter" {
  if (!data.length) return "bar";

  const xSample = data[0][xKey];
  const ySample = data[0][yKeys[0]];

  // Both numeric → scatter
  if (typeof xSample === "number" && typeof ySample === "number") {
    return "scatter";
  }

  // Many distinct x values (time-series feel) → line
  const distinctX = new Set(data.map((d) => d[xKey])).size;
  if (distinctX > 10) return "line";

  return "bar";
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

interface TooltipEntry {
  name: string;
  value: unknown;
  color?: string;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-[6px] px-3 py-2"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
        boxShadow: "rgba(0,0,0,0.1) 0px 4px 12px",
        fontFamily: "system-ui",
      }}
    >
      {label && (
        <p
          style={{
            fontSize: "11px",
            fontWeight: 500,
            color: "rgba(38,37,30,0.55)",
            marginBottom: "4px",
          }}
        >
          {label}
        </p>
      )}
      {payload.map((entry, i) => (
        <p key={i} style={{ fontSize: "12px", color: entry.color ?? "#26251e" }}>
          {entry.name}:{" "}
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontWeight: 500,
            }}
          >
            {String(entry.value)}
          </span>
        </p>
      ))}
    </div>
  );
}

// ── Axis tick styles ──────────────────────────────────────────────────────────

const TICK_STYLE = {
  fontFamily: "system-ui",
  fontSize: 11,
  fill: "rgba(38,37,30,0.45)",
};

// ── Chart variants ────────────────────────────────────────────────────────────

function BarChartView({
  data,
  xKey,
  yKeys,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} barGap={4}>
        <CartesianGrid vertical={false} stroke="rgba(38,37,30,0.06)" />
        <XAxis
          dataKey={xKey}
          axisLine={false}
          tickLine={false}
          tick={TICK_STYLE}
        />
        <YAxis axisLine={false} tickLine={false} tick={TICK_STYLE} width={40} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(38,37,30,0.04)" }} />
        {yKeys.length > 1 && <Legend wrapperStyle={TICK_STYLE} />}
        {yKeys.map((key, i) => (
          <Bar
            key={key}
            dataKey={key}
            fill={AI_COLORS[i % AI_COLORS.length]}
            radius={[3, 3, 0, 0]}
            barSize={yKeys.length > 1 ? undefined : 32}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function LineChartView({
  data,
  xKey,
  yKeys,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data}>
        <CartesianGrid stroke="rgba(38,37,30,0.06)" />
        <XAxis
          dataKey={xKey}
          axisLine={false}
          tickLine={false}
          tick={TICK_STYLE}
        />
        <YAxis axisLine={false} tickLine={false} tick={TICK_STYLE} width={40} />
        <Tooltip content={<CustomTooltip />} />
        {yKeys.length > 1 && <Legend wrapperStyle={TICK_STYLE} />}
        {yKeys.map((key, i) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={AI_COLORS[i % AI_COLORS.length]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function ScatterChartView({
  data,
  xKey,
  yKeys,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart>
        <CartesianGrid stroke="rgba(38,37,30,0.06)" />
        <XAxis
          dataKey={xKey}
          name={xKey}
          axisLine={false}
          tickLine={false}
          tick={TICK_STYLE}
          type="number"
        />
        <YAxis
          dataKey={yKeys[0]}
          name={yKeys[0]}
          axisLine={false}
          tickLine={false}
          tick={TICK_STYLE}
          width={40}
        />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<CustomTooltip />} />
        <Scatter data={data} fill={AI_COLORS[0]} opacity={0.6} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ChartRenderer({
  data,
  xKey,
  yKeys,
  chartType = "auto",
  height = 240,
  title,
}: ChartRendererProps) {
  if (!data.length) return null;

  const cols = Object.keys(data[0]);
  const resolvedX = xKey ?? cols[0];
  const resolvedY =
    yKeys ??
    cols
      .filter((c) => c !== resolvedX && typeof data[0][c] === "number")
      .slice(0, 3);

  if (!resolvedY.length) return null;

  const resolvedType =
    chartType === "auto"
      ? detectChartType(data, resolvedX, resolvedY)
      : chartType;

  return (
    <div>
      {title && (
        <p
          className="mb-3"
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.048px",
            color: "rgba(38,37,30,0.45)",
          }}
        >
          {title}
        </p>
      )}
      <div style={{ height }}>
        {resolvedType === "scatter" ? (
          <ScatterChartView data={data} xKey={resolvedX} yKeys={resolvedY} />
        ) : resolvedType === "line" ? (
          <LineChartView data={data} xKey={resolvedX} yKeys={resolvedY} />
        ) : (
          <BarChartView data={data} xKey={resolvedX} yKeys={resolvedY} />
        )}
      </div>
    </div>
  );
}
