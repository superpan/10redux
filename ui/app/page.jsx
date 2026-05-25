"use client";

import { useEffect, useRef, useState } from "react";

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

export default function Page() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [playing, setPlaying] = useState(null);
  const [libraries, setLibraries] = useState([]);
  const [library, setLibrary] = useState(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem("ten.library") || "";
  });

  useEffect(() => {
    fetch("/stats").then((r) => r.json()).then(setStats).catch(() => {});
    fetch("/libraries")
      .then((r) => r.json())
      .then((d) => setLibraries(d.libraries || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("ten.library", library);
    }
  }, [library]);

  async function doSearch(e) {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ q: query, limit: "24" });
      if (library) params.set("library", library);
      const r = await fetch(`/search?${params.toString()}`);
      const data = await r.json();
      setHits(data.hits || []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>ten</h1>
        <span className="stats">
          {stats?.ten_visual?.points != null
            ? `${stats.ten_visual.points} clips indexed`
            : "qdrant offline"}
        </span>
      </header>

      <form className="search" onSubmit={doSearch}>
        <input
          autoFocus
          placeholder="describe what you're looking for…  (e.g. 'a person catching a ball at sunset')"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {libraries.length > 0 && (
          <select
            value={library}
            onChange={(e) => setLibrary(e.target.value)}
            title="Scope search to a single library (parent dir of indexed videos)"
          >
            <option value="">all libraries</option>
            {libraries.map((lib) => (
              <option key={lib} value={lib}>
                {lib}
              </option>
            ))}
          </select>
        )}
        <button type="submit" disabled={loading}>
          {loading ? "searching…" : "search"}
        </button>
      </form>

      {hits.length === 0 ? (
        <div className="empty">{loading ? "searching…" : "no results yet — try a query above"}</div>
      ) : (
        <div className="grid">
          {hits.map((h) => (
            <Card key={h.clip_id} hit={h} onPlay={setPlaying} />
          ))}
        </div>
      )}

      {playing && <Player clip={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}

function Card({ hit, onPlay }) {
  const p = hit.payload;
  const [summary, setSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  async function loadSummary() {
    if (summary || loadingSummary) return;
    setLoadingSummary(true);
    try {
      const r = await fetch(`/clip/${p.clip_id}/summary`);
      const data = await r.json();
      setSummary(data.summary);
    } catch {
      setSummary("(failed to summarize)");
    } finally {
      setLoadingSummary(false);
    }
  }

  return (
    <div className="card">
      <img
        className="thumb"
        src={`/thumb/${p.clip_id}.jpg`}
        alt=""
        onClick={() =>
          onPlay({ clip_id: p.clip_id, t_start: p.t_start, name: p.video_name })
        }
      />
      <div className="body">
        <div className="meta">
          <span className="src">{hit.sources.join("+")}</span>
          <span>score {hit.score.toFixed(3)}</span>
          <span style={{ marginLeft: "auto" }}>
            {fmtTime(p.t_start)}–{fmtTime(p.t_end)}
          </span>
        </div>
        <div className="name" title={p.video_path}>
          {p.video_name}
        </div>
        <div className="caption">{p.caption}</div>
        <div className="actions">
          <button
            onClick={() =>
              onPlay({ clip_id: p.clip_id, t_start: p.t_start, name: p.video_name })
            }
          >
            ▶ play
          </button>
          <button onClick={loadSummary} disabled={loadingSummary}>
            {loadingSummary ? "summarizing…" : summary ? "resummarize" : "summarize"}
          </button>
        </div>
        {(loadingSummary || summary) && (
          <div className={`summary${loadingSummary ? " loading" : ""}`}>
            {loadingSummary ? "Qwen3-VL is summarizing this clip…" : summary}
          </div>
        )}
      </div>
    </div>
  );
}

function Player({ clip, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    const onMeta = () => {
      v.currentTime = clip.t_start;
      v.play().catch(() => {});
    };
    v.addEventListener("loadedmetadata", onMeta);
    return () => v.removeEventListener("loadedmetadata", onMeta);
  }, [clip]);

  return (
    <div className="modal-bg" onClick={onClose}>
      <button className="close" onClick={onClose}>
        close
      </button>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <video ref={ref} src={`/clip/${clip.clip_id}/stream`} controls preload="metadata" />
      </div>
    </div>
  );
}
