import VideoAnalysis from "./modules/VideoAnalysis/index.jsx";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>StatScout</h1>
        <span className="badge">Video Analysis</span>
      </header>
      <main>
        <VideoAnalysis />
      </main>
    </div>
  );
}
