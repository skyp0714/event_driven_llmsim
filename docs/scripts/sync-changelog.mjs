// Mirror the repo-root CHANGELOG.md into a Docusaurus page at /changelog.
//
// Run automatically by `pnpm start` / `pnpm build` via the prestart /
// prebuild hooks in package.json. The output (`src/pages/changelog.md`) is
// committed so the page works out of the box on a fresh clone too; the
// hooks just keep it in sync with the source of truth before each build.

import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const src = resolve(repoRoot, 'CHANGELOG.md');
const dst = resolve(here, '../src/pages/changelog.md');

const body = readFileSync(src, 'utf-8');

const frontmatter = `---
title: Changelog
description: LLMServingSim release history
---

`;

// CHANGELOG.md already starts with a top-level "# Changelog" heading; strip
// the duplicate that Docusaurus would otherwise prepend from the title.
const trimmed = body.replace(/^# Changelog\s*\n+/, '');

writeFileSync(dst, frontmatter + trimmed);

console.log(`[sync-changelog] wrote ${dst} (${trimmed.length} chars)`);
