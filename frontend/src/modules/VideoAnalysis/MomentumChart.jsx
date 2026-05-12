/**
 * Match-momentum timeline rendered as an SVG.
 *
 * Each sample is { t: seconds, advantage: -1..+1 }.
 *   advantage > 0  → Team A held the ball through the rolling window (drawn above center, cyan)
 *   advantage < 0  → Team B held the ball (drawn below center, pink)
 *
 * Visual style: a smoothly-stroked baseline with a filled momentum river that
 * flips color depending on which side of zero the curve is on. Inspired by
 * broadcast-graphics momentum bars.
 */
export default function MomentumChart({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return <p className="momentum-empty">Not enough possession data to chart momentum.</p>;
  }

  const W = 1000;
  const H = 220;
  const PAD_X = 40;
  const PAD_Y = 18;
  const innerW = W - PAD_X * 2;
  const innerH = H - PAD_Y * 2;
  const centerY = H / 2;

  const tMax = Math.max(1, timeline[timeline.length - 1].t);

  const xOf = (t) => PAD_X + (t / tMax) * innerW;
  const yOf = (a) => centerY - a * (innerH / 2);

  // Build the curve (smooth via polyline since samples are dense)
  const linePath = timeline
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.t).toFixed(1)} ${yOf(p.advantage).toFixed(1)}`)
    .join(" ");

  // Build two separate filled regions: one for advantage>0 (Team A), one for <0 (Team B).
  // We clip each region against the centerline so they only show their own half.
  const polyAFill =
    `M ${xOf(timeline[0].t)} ${centerY} ` +
    timeline.map((p) => `L ${xOf(p.t).toFixed(1)} ${yOf(p.advantage).toFixed(1)}`).join(" ") +
    ` L ${xOf(timeline[timeline.length - 1].t)} ${centerY} Z`;

  // Tick marks every 5 seconds
  const ticks = [];
  for (let s = 0; s <= tMax; s += 5) {
    ticks.push(s);
  }

  return (
    <div className="momentum-chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="momentum-svg">
        <defs>
          <linearGradient id="teamAGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00BFFF" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#00BFFF" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="teamBGrad" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="#EC4899" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#EC4899" stopOpacity="0" />
          </linearGradient>
          {/* clip top half (above centerline) — Team A region */}
          <clipPath id="topClip">
            <rect x="0" y="0" width={W} height={centerY} />
          </clipPath>
          {/* clip bottom half (below centerline) — Team B region */}
          <clipPath id="bottomClip">
            <rect x="0" y={centerY} width={W} height={H - centerY} />
          </clipPath>
        </defs>

        {/* Grid: dashed centerline */}
        <line
          x1={PAD_X}
          x2={W - PAD_X}
          y1={centerY}
          y2={centerY}
          stroke="#CBD5E1"
          strokeWidth="1"
          strokeDasharray="4,5"
        />

        {/* Tick labels */}
        {ticks.map((s) => (
          <g key={s}>
            <line
              x1={xOf(s)}
              x2={xOf(s)}
              y1={H - PAD_Y}
              y2={H - PAD_Y + 4}
              stroke="#94A3B8"
              strokeWidth="1"
            />
            <text
              x={xOf(s)}
              y={H - 2}
              textAnchor="middle"
              fontSize="11"
              fill="#64748B"
            >
              {s}s
            </text>
          </g>
        ))}

        {/* Filled regions (clipped halves stack to give the bicolor effect) */}
        <g clipPath="url(#topClip)">
          <path d={polyAFill} fill="url(#teamAGrad)" />
        </g>
        <g clipPath="url(#bottomClip)">
          <path d={polyAFill} fill="url(#teamBGrad)" />
        </g>

        {/* The curve itself */}
        <path
          d={linePath}
          fill="none"
          stroke="#0E1838"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Y-axis hints */}
        <text x={PAD_X - 8} y={PAD_Y + 8} textAnchor="end" fontSize="10" fill="#0EA5E9" fontWeight="600">
          TEAM A
        </text>
        <text x={PAD_X - 8} y={H - PAD_Y - 2} textAnchor="end" fontSize="10" fill="#EC4899" fontWeight="600">
          TEAM B
        </text>
      </svg>
    </div>
  );
}
