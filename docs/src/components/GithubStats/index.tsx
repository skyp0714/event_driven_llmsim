import {useEffect, useState, type ReactNode} from 'react';
import {Star, GitFork, Eye} from 'lucide-react';
import styles from './styles.module.css';

const REPO = 'casys-kaist/LLMServingSim';
const API_URL = `https://api.github.com/repos/${REPO}`;
const REPO_URL = `https://github.com/${REPO}`;
const CACHE_KEY = 'gh-stats:LLMServingSim';
const CACHE_TTL_MS = 5 * 60 * 1000;

type Stats = {
  stars: number;
  forks: number;
  watchers: number;
};

type Cached = {
  data: Stats;
  ts: number;
};

function format(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function readCache(): Stats | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Cached;
    if (Date.now() - parsed.ts > CACHE_TTL_MS) return null;
    return parsed.data;
  } catch {
    return null;
  }
}

function writeCache(data: Stats): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({data, ts: Date.now()}),
    );
  } catch {
    // ignore quota errors
  }
}

export default function GithubStats(): ReactNode {
  const [stats, setStats] = useState<Stats | null>(() => readCache());
  const [error, setError] = useState(false);

  useEffect(() => {
    if (stats) return;
    let cancelled = false;
    fetch(API_URL, {headers: {Accept: 'application/vnd.github+json'}})
      .then((res) => {
        if (!res.ok) throw new Error(`GitHub API ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const next: Stats = {
          stars: data.stargazers_count ?? 0,
          forks: data.forks_count ?? 0,
          // GitHub API quirk: `watchers_count` mirrors stars; the real
          // watch / subscriber count is `subscribers_count`.
          watchers: data.subscribers_count ?? 0,
        };
        setStats(next);
        writeCache(next);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [stats]);

  // Fail silently, if the API is rate-limited or down, skip the row
  // rather than showing a broken state on the homepage.
  if (error || !stats) {
    return <div className={styles.placeholder} aria-hidden="true" />;
  }

  return (
    <div className={styles.row} aria-label="GitHub repository stats">
      <a
        className={styles.item}
        href={`${REPO_URL}/stargazers`}
        target="_blank"
        rel="noopener noreferrer">
        <Star size={18} strokeWidth={2} />
        <span className={styles.value}>{format(stats.stars)}</span>
        <span className={styles.label}>Stars</span>
      </a>
      <a
        className={styles.item}
        href={`${REPO_URL}/network/members`}
        target="_blank"
        rel="noopener noreferrer">
        <GitFork size={18} strokeWidth={2} />
        <span className={styles.value}>{format(stats.forks)}</span>
        <span className={styles.label}>Forks</span>
      </a>
      <a
        className={styles.item}
        href={`${REPO_URL}/watchers`}
        target="_blank"
        rel="noopener noreferrer">
        <Eye size={18} strokeWidth={2} />
        <span className={styles.value}>{format(stats.watchers)}</span>
        <span className={styles.label}>Watchers</span>
      </a>
    </div>
  );
}
