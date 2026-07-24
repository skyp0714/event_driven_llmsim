import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import type {LucideIcon} from 'lucide-react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

export type IconCard = {
  Icon: LucideIcon;
  title: string;
  description: ReactNode;
  /** Internal docs link */
  to?: string;
  /** External link (opens in new tab) */
  href?: string;
};

type Props = {
  cards: IconCard[];
  columns?: 2 | 3 | 4;
};

function CardBody({Icon, title, description}: IconCard) {
  return (
    <>
      <div className={styles.iconBadge} aria-hidden="true">
        <Icon size={22} strokeWidth={1.75} />
      </div>
      <Heading as="h3" className={styles.title}>
        {title}
      </Heading>
      <p className={styles.description}>{description}</p>
    </>
  );
}

export default function IconCardGrid({
  cards,
  columns = 3,
}: Props): ReactNode {
  const colClass = {
    2: styles.cols2,
    3: styles.cols3,
    4: styles.cols4,
  }[columns];

  return (
    <div className={clsx(styles.grid, colClass)}>
      {cards.map((card, i) => {
        const inner = <CardBody {...card} />;
        if (card.to) {
          return (
            <Link key={i} className={styles.card} to={card.to}>
              {inner}
            </Link>
          );
        }
        if (card.href) {
          return (
            <a
              key={i}
              className={styles.card}
              href={card.href}
              target="_blank"
              rel="noopener noreferrer">
              {inner}
            </a>
          );
        }
        return (
          <div key={i} className={styles.card}>
            {inner}
          </div>
        );
      })}
    </div>
  );
}
