import MomentumChart from "./MomentumChart.jsx";

/**
 * High-level match statistics block.
 *  - Possession bar (Team A vs Team B share of frames where a player held the ball)
 *  - Match momentum chart (rolling possession over time)
 *
 * Per-player speed table is intentionally removed — speed is shown directly on
 * each player in the broadcast video as a clean inline label.
 */
export default function StatsPanel({ analytics }) {
  if (!analytics) {
    return <p className="va-empty">No analytics data available for this clip yet.</p>;
  }

  const { possession, momentum_timeline } = analytics;
  const teamA = possession?.team_a_pct ?? 50;
  const teamB = possession?.team_b_pct ?? 50;

  return (
    <div className="va-stats-stack">
      <section className="va-card va-stat-card">
        <header className="va-stat-header">
          <h3>Ball Possession</h3>
          <p className="va-muted">Share of frames where each team had the closest player to the ball</p>
        </header>

        <div className="va-possession-bar">
          <div
            className="va-poss-fill va-poss-a"
            style={{ width: `${teamA}%` }}
            aria-label={`Team A possession ${teamA}%`}
          >
            {teamA >= 8 && <span>{teamA}%</span>}
          </div>
          <div
            className="va-poss-fill va-poss-b"
            style={{ width: `${teamB}%` }}
            aria-label={`Team B possession ${teamB}%`}
          >
            {teamB >= 8 && <span>{teamB}%</span>}
          </div>
        </div>

        <div className="va-poss-legend">
          <span><i className="va-swatch va-swatch-a" /> Team A — {teamA}%</span>
          <span><i className="va-swatch va-swatch-b" /> Team B — {teamB}%</span>
        </div>
      </section>

      <section className="va-card va-stat-card">
        <header className="va-stat-header">
          <h3>Match Momentum</h3>
          <p className="va-muted">
            Rolling 3-second possession advantage across the clip — above the
            centerline means Team A is in control, below means Team B.
          </p>
        </header>
        <MomentumChart timeline={momentum_timeline} />
      </section>
    </div>
  );
}
