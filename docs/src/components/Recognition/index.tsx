import type {ReactNode, ComponentType, SVGProps} from 'react';
import {Trophy, Medal, ScrollText} from 'lucide-react';
import styles from './styles.module.css';

type Variant = 'pub' | 'best' | 'artifact';
type IconType = ComponentType<SVGProps<SVGSVGElement> & {size?: number; strokeWidth?: number}>;

type Stat = {
  count: string;
  label: string;
  variant: Variant;
  Icon: IconType;
};

type Badge = {
  label: string;
  variant: Exclude<Variant, 'pub'>;
};

const STATS: Stat[] = [
  {count: '3', label: 'Publications', variant: 'pub', Icon: ScrollText},
  {count: '2', label: 'Best Paper Awards', variant: 'best', Icon: Trophy},
  {count: '1', label: 'Distinguished Artifact', variant: 'artifact', Icon: Medal},
];

const AWARDS: {venue: string; href: string; badges: Badge[]}[] = [
  {
    venue: 'IISWC 2024',
    href: 'https://doi.org/10.1109/IISWC63097.2024.00012',
    badges: [
      {label: 'Best Paper Award', variant: 'best'},
      {label: 'Distinguished Artifact Award', variant: 'artifact'},
    ],
  },
  {
    venue: 'ISPASS 2026',
    href: 'https://github.com/casys-kaist/LLMServingSim#publications',
    badges: [{label: 'Best Paper Award', variant: 'best'}],
  },
];

const BADGE_ICONS: Record<Exclude<Variant, 'pub'>, IconType> = {
  best: Trophy,
  artifact: Medal,
};

export default function Recognition(): ReactNode {
  return (
    <section className={styles.recognition} aria-labelledby="recognition-heading">
      <div className="container">
        <p className={styles.eyebrow}>Recognition</p>
        <h2 id="recognition-heading" className={styles.title}>
          Three publications. Three awards.
        </h2>

        <div className={styles.stats}>
          {STATS.map(({count, label, variant, Icon}) => (
            <div key={label} className={`${styles.stat} ${styles[`stat_${variant}`]}`}>
              <div className={styles.statIcon} aria-hidden="true">
                <Icon size={20} strokeWidth={2} />
              </div>
              <div className={styles.count}>{count}</div>
              <div className={styles.label}>{label}</div>
            </div>
          ))}
        </div>

        <ul className={styles.awards}>
          {AWARDS.map(({venue, href, badges}) => (
            <li key={venue} className={styles.awardRow}>
              <a className={styles.venue} href={href} target="_blank" rel="noopener noreferrer">
                {venue}
              </a>
              <div className={styles.badges}>
                {badges.map(({label, variant}) => {
                  const Icon = BADGE_ICONS[variant];
                  return (
                    <span key={label} className={`${styles.badge} ${styles[variant]}`}>
                      <Icon size={16} strokeWidth={2.25} aria-hidden="true" />
                      <span>{label}</span>
                    </span>
                  );
                })}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
