import { useEffect, useRef } from "react";
import StatsPanel from "./StatsPanel.jsx";

const API_HOST = "http://localhost:8000";

/**
 * Match-analysis results page styled to match the rest of the app
 * (cream background, navy display headers, white cards, red-accented copy).
 *
 * Layout:
 *   ── Page heading
 *   ── Broadcast video        (left, big)         │ Tactical 2D pitch (top)
 *                                                 │ Territorial control (bottom)
 *   ── Description card explaining tactical view
 *   ── Description card explaining territorial control
 *   ── Stats block (possession + momentum chart)
 */
export default function ResultsViewer({
  videoUrl,
  mainVideoUrl,
  pitchVideoUrl,
  voronoiVideoUrl,
  analytics,
}) {
  const mainRef = useRef(null);
  const pitchRef = useRef(null);
  const voronoiRef = useRef(null);

  const mainSrc = `${API_HOST}${mainVideoUrl || videoUrl}`;
  const pitchSrc = pitchVideoUrl ? `${API_HOST}${pitchVideoUrl}` : null;
  const voronoiSrc = voronoiVideoUrl ? `${API_HOST}${voronoiVideoUrl}` : null;
  const hasSidePanels = Boolean(pitchSrc && voronoiSrc);

  // Master video controls drive the side panels; small drift correction keeps them aligned.
  useEffect(() => {
    const main = mainRef.current;
    if (!main || !hasSidePanels) return;
    const sides = [pitchRef.current, voronoiRef.current].filter(Boolean);

    const syncPlay = () => sides.forEach((v) => v.play().catch(() => {}));
    const syncPause = () => sides.forEach((v) => v.pause());
    const syncSeek = () => sides.forEach((v) => { v.currentTime = main.currentTime; });
    const syncRate = () => sides.forEach((v) => { v.playbackRate = main.playbackRate; });
    const driftFix = () => {
      sides.forEach((v) => {
        if (Math.abs(v.currentTime - main.currentTime) > 0.15) {
          v.currentTime = main.currentTime;
        }
      });
    };

    main.addEventListener("play", syncPlay);
    main.addEventListener("pause", syncPause);
    main.addEventListener("seeked", syncSeek);
    main.addEventListener("ratechange", syncRate);
    main.addEventListener("timeupdate", driftFix);

    return () => {
      main.removeEventListener("play", syncPlay);
      main.removeEventListener("pause", syncPause);
      main.removeEventListener("seeked", syncSeek);
      main.removeEventListener("ratechange", syncRate);
      main.removeEventListener("timeupdate", driftFix);
    };
  }, [hasSidePanels, mainSrc, pitchSrc, voronoiSrc]);

  return (
    <div className="va-page">
      <header className="va-pageheader">
        <h1>Match Analysis</h1>
        <p>Live broadcast · Tactical pitch · Territorial control</p>
      </header>

      <div className={`va-grid ${hasSidePanels ? "with-panels" : "no-panels"}`}>
        <div className="va-card va-broadcast">
          <div className="va-card-label">
            <span className="va-dot va-dot-live" />
            BROADCAST
          </div>
          <div className="va-video">
            <video ref={mainRef} src={mainSrc} controls playsInline />
          </div>
        </div>

        {hasSidePanels && (
          <>
            <div className="va-card va-tactical">
              <div className="va-card-label">
                <span className="va-dot va-dot-blue" />
                TACTICAL VIEW
              </div>
              <div className="va-video">
                <video ref={pitchRef} src={pitchSrc} muted playsInline />
              </div>
            </div>

            <div className="va-card va-territorial">
              <div className="va-card-label">
                <span className="va-dot va-dot-pink" />
                TERRITORIAL CONTROL
              </div>
              <div className="va-video">
                <video ref={voronoiRef} src={voronoiSrc} muted playsInline />
              </div>
            </div>
          </>
        )}
      </div>

      {hasSidePanels && (
        <>
          <div className="va-description">
            <span className="va-star">✦</span>
            <p>
              <strong>Tactical View </strong> projects every detected player onto a
              top-down pitch using <em>homography</em> — a mathematical mapping
              between the broadcast camera and the real field, computed each frame
              from the visible pitch markings. The result is a clean 2D map that
              exposes formations, lines, and runs that are hard to read from the
              broadcast angle.
            </p>
          </div>

          <div className="va-description">
            <span className="va-star">✦</span>
            <p>
              <strong>Territorial Control </strong> assigns every pixel of the
              pitch to whichever team has the closest player, then smooths the
              boundary using a tanh blend so the zones flow rather than tile. Cyan
              areas are space dominated by Team A, pink by Team B — this is the
              same visualization elite analytics teams use to study how a side
              compresses the opposition or stretches it on the counter.
            </p>
          </div>
        </>
      )}

      <StatsPanel analytics={analytics} />
    </div>
  );
}
