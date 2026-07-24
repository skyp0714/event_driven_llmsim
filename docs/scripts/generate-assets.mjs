/**
 * Generate favicon + apple-touch-icon + social card from the source logo PNGs.
 *
 * Run from website/:
 *   node scripts/generate-assets.mjs
 *
 * Inputs (in website/static/img/):
 *   - llmservingsim_compact_primary_transparent.png
 *   - llmservingsim_full_primary_transparent.png
 *
 * Outputs (in website/static/img/):
 *   - favicon.png            32×32
 *   - favicon-256.png        256×256 (high-DPI fallback)
 *   - apple-touch-icon.png   180×180 on white
 *   - social-card.png        1200×630 on indigo (OG / Twitter card)
 */

import sharp from 'sharp';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATIC_IMG = path.join(__dirname, '..', 'static', 'img');

const COMPACT = path.join(STATIC_IMG, 'llmservingsim_compact_primary_transparent.png');
const FULL = path.join(STATIC_IMG, 'llmservingsim_full_primary_transparent.png');

// Fit `src` into a `size`×`size` canvas with padding ratio.
async function compactSquare({src, size, paddingRatio, background}) {
  const inner = Math.round(size * (1 - paddingRatio * 2));
  const fitted = await sharp(src)
    .resize(inner, inner, {fit: 'inside', withoutEnlargement: false})
    .toBuffer();
  return sharp({
    create: {
      width: size,
      height: size,
      channels: 4,
      background,
    },
  })
    .composite([{input: fitted, gravity: 'center'}])
    .png()
    .toBuffer();
}

async function generateFavicon32() {
  const buf = await compactSquare({
    src: COMPACT,
    size: 32,
    paddingRatio: 0.05,
    background: {r: 0, g: 0, b: 0, alpha: 0},
  });
  await sharp(buf).toFile(path.join(STATIC_IMG, 'favicon.png'));
  console.log('✓ favicon.png (32×32)');
}

async function generateFavicon256() {
  const buf = await compactSquare({
    src: COMPACT,
    size: 256,
    paddingRatio: 0.08,
    background: {r: 0, g: 0, b: 0, alpha: 0},
  });
  await sharp(buf).toFile(path.join(STATIC_IMG, 'favicon-256.png'));
  console.log('✓ favicon-256.png (256×256)');
}

async function generateAppleTouchIcon() {
  const buf = await compactSquare({
    src: COMPACT,
    size: 180,
    paddingRatio: 0.12,
    background: {r: 255, g: 255, b: 255, alpha: 1},
  });
  await sharp(buf).toFile(path.join(STATIC_IMG, 'apple-touch-icon.png'));
  console.log('✓ apple-touch-icon.png (180×180, white bg)');
}

async function generateSocialCard() {
  const W = 1200;
  const H = 630;

  // Background: white with a soft indigo radial glow at top-center.
  // Whole-canvas SVG so the gradient covers the entire card without
  // overflow / negative-offset issues.
  const bgSvg = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
       <defs>
         <radialGradient id="g" cx="50%" cy="0%" r="60%">
           <stop offset="0%" stop-color="#6366f1" stop-opacity="0.28"/>
           <stop offset="60%" stop-color="#6366f1" stop-opacity="0.05"/>
           <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
         </radialGradient>
       </defs>
       <rect width="${W}" height="${H}" fill="white"/>
       <rect width="${W}" height="${H}" fill="url(#g)"/>
     </svg>`,
  );

  // Wordmark sized to the card width with breathing room
  const targetWidth = Math.round(W * 0.7);
  const wordmarkResized = await sharp(FULL)
    .resize({width: targetWidth, fit: 'inside'})
    .toBuffer();
  const wordmarkMeta = await sharp(wordmarkResized).metadata();

  // Tagline as SVG (system sans-serif fallback)
  const taglineSvg = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="80">
       <style>
         .t { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              font-size: 28px; font-weight: 500; fill: #475569; letter-spacing: -0.3px; }
       </style>
       <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" class="t">
         Toward unified simulation of heterogeneous and disaggregated LLM serving infrastructure
       </text>
     </svg>`,
  );

  // Vertical layout: wordmark slightly above center, tagline below.
  const wordmarkTop = Math.round((H - wordmarkMeta.height) / 2 - 40);
  const wordmarkLeft = Math.round((W - wordmarkMeta.width) / 2);
  const taglineTop = wordmarkTop + wordmarkMeta.height + 40;

  await sharp(bgSvg)
    .composite([
      {input: wordmarkResized, top: wordmarkTop, left: wordmarkLeft},
      {input: taglineSvg, top: taglineTop, left: 0},
    ])
    .png()
    .toFile(path.join(STATIC_IMG, 'social-card.png'));
  console.log('✓ social-card.png (1200×630, white bg + indigo glow)');
}

async function main() {
  await generateFavicon32();
  await generateFavicon256();
  await generateAppleTouchIcon();
  await generateSocialCard();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
