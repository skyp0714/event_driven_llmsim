import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import GithubStats from '@site/src/components/GithubStats';
import Recognition from '@site/src/components/Recognition';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const wordmark = useBaseUrl(
    '/img/llmservingsim_full_primary_transparent.png',
  );
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img
          src={wordmark}
          alt={siteConfig.title}
          className={styles.heroWordmark}
        />
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <GithubStats />
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started/overview">
            Get Started
          </Link>
          <Link
            className="button button--secondary button--lg"
            href="https://github.com/casys-kaist/LLMServingSim">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Toward unified simulation of heterogeneous and disaggregated LLM serving infrastructure">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <Recognition />
      </main>
    </Layout>
  );
}
