---
author: ["Abdullah Al Mamun"]
title: "Personal Portfolio Audit + Strategy Plan"
date: 2026-05-13
draft: true
comments: false
ShowToc: true
TocOpen: true
math: false
description: "End-to-end audit of newabdullah.com plus a strategy plan to attract VCs, founders, and recruiters; grow a learner audience; and run a cross-platform publishing pipeline."
summary: "Audit findings, benchmarks against top AI researcher portfolios, prioritized recommendations split by audience, and a concrete POSSE (Post Once, Syndicate Everywhere) pipeline."
tags:
  - "portfolio"
  - "seo"
  - "content strategy"
  - "personal brand"
  - "blogging"
  - "posse"
categories:
  - "Meta"
---

> **Draft — local reference only.** This post is marked `draft: true` so it is excluded from the production build (`hugo --gc --minify` skips drafts). View locally with `hugo server -D`.

Long but actionable. Split into 5 parts:
**(1)** What's broken right now (audit), **(2)** what top AI researchers actually do, **(3)** recommendations to attract VCs/founders/recruiters, **(4)** recommendations to grow a learner audience, **(5)** cross-posting pipeline (POSSE).

---

## Part 1 — Current Audit (what I found)

### 1.1 SEO — Major gaps

| Item | Status | Impact |
|---|---|---|
| `<title>` on home | ❌ Just "Abdullah Al Mamun" — no keywords | Lose search traffic for "ML engineer Atlassian", "RecSys blog", etc. |
| `meta description` on home | ❌ **Missing entirely** | Google shows random text snippet; click-through rate drops ~30% |
| `og:image` (social card image) | ❌ Missing on every page | Links shared on LinkedIn/X/Discord show NO preview image → ignored |
| Twitter card meta | ❌ Missing | Same as above on X |
| Canonical URL | ❌ Missing | SEO duplicate-content risk (esp. with dev URL also live) |
| JSON-LD `Person` schema | ❌ Missing on home | No knowledge-graph eligibility, no rich Google "people also ask" |
| Post `meta description` | ❌ Missing on every post | Same CTR loss per post |
| Post `og:image` | ❌ Missing on every post | No social previews when posts are shared |
| URL slug consistency | ⚠️ "the-evaluation-of-recsys**---**part-1" (triple-dash) | Looks broken, hurts shareability |
| robots.txt + sitemap + RSS | ✅ Present | Good baseline |
| `article:published_time` on posts | ✅ Present | Good |

### 1.2 Content inventory — Thin

| Section | Have | Missing |
|---|---|---|
| Posts | 3 (all RecSys series) | Diversity (LLMs, agents, career, retrospectives) |
| About | ✅ Comprehensive, just updated | Quantified impact ("$xxxM" placeholder still in long bio) |
| Resume | ✅ PDF | No HTML version (Google can't index PDFs as well) |
| Contact | ✅ Form | Good |
| Talks | ❌ | High-credibility signal missing |
| Publications | ❌ | You have a PhD with papers — **huge missed signal** for credibility |
| Projects / Open Source | ❌ | No portfolio of "things I've built" |
| Newsletter / Subscribe | ⚠️ Follow.it form only on post pages | Not on home — biggest leak |
| Now / Reading list | ❌ | Personal flavor missing |
| Comments | ⚠️ Frontmatter says `comments: true` but no Giscus/Disqus config wired | Engagement loop broken |
| Internal linking between posts | ⚠️ Sparse | Bad for SEO and dwell time |

### 1.3 Performance — Good

- Pages are tiny (7–23KB). No tracking pixels. No bloat. Hugo did its job. ✅
- Submodule theme is current. ✅
- HTTPS enforced, valid cert. ✅
- One thing missing: **no Google Analytics, no Google Search Console**, so you're flying blind on what works.

---

## Part 2 — What top AI researchers actually do

I'll reference specific peers you should benchmark against (open them side-by-side):

| Person | URL | What they do well |
|---|---|---|
| **Lilian Weng** | `lilianweng.github.io` | **The gold standard.** 10-20k word deep dives on transformers, RL, agents. Sidebar TOC. Tags. Citations. Math. Posts become canonical references. |
| **Andrej Karpathy** | `karpathy.ai` + `karpathy.github.io/blog` | Minimal design. Long-form posts + linked YouTube videos. GitHub-first projects. Talks list. |
| **Chip Huyen** | `huyenchip.com` | Mix of technical, career, book promo. Newsletter prominent. Book/course CTAs. Photography page (humanizes). |
| **Sebastian Raschka** | `sebastianraschka.com` + Substack | Substack-first ($$$). Free + paid tiers. Magazine format. Builds in public. |
| **Eugene Yan** | `eugeneyan.com` | Industry ML focus. Posts have TLDR. Tags. Newsletter. "Now" page. Reading list. Very practical applied content. |
| **Simon Willison** | `simonwillison.net` | Daily microblog + TIL (Today I Learned) + Atom feeds for every tag. ~5000 posts. |
| **Jay Alammar** | `jalammar.github.io` | Visual storytelling with custom illustrations ("Illustrated Transformer"). Rare but viral posts. |
| **Hamel Husain** | `hamel.dev` | Practitioner LLM evals + RAG. Honest opinions. Drives consulting. |

**Cross-cutting patterns:**

1. **Long-form deep dives become reference material** — one good post can drive years of traffic.
2. **Series with clear arc** (you've started this with "Evaluation of RecSys Part 1/2/3" — good instinct).
3. **TL;DR / summary** at the top of every post.
4. **Tags as first-class** — discoverability is everything.
5. **Newsletter is the moat** — Google traffic is fickle; email is owned.
6. **GitHub-first**: post + companion repo with runnable code.
7. **"Now" page** ("What I'm working on, reading, building right now") — humanizes you.
8. **Visual diagrams** (Excalidraw, Figma) — drastically boost shares.
9. **Cross-posted with canonical pointing home** — POSSE pattern.
10. **No business cruft** — no popups, no autoplay, no ads, no AI chatbots.

---

## Part 3 — Recommendations to attract VCs, founders, recruiters

The #1 question they have when they land: **"What can this person uniquely do for me?"** Make the answer obvious in 5 seconds.

### Tier 1 (do this week — high impact, low effort)

| # | Action | Why |
|---|---|---|
| 1 | Add `meta description`, `og:image`, Twitter card to every page | Every link you ever share on LinkedIn/X will look 10x more professional |
| 2 | Create a single "social card" image (1200×630 PNG, just your name + tagline + photo) and use it as default `og:image` | One asset solves the problem above |
| 3 | Above-the-fold hero on home: rewrite to "**I build Agentic AI for enterprise search at Atlassian.** Ex-Meta Ads ML. PhD in interpretable deep learning. I write about RecSys, RL, and applied LLMs." | Founder-style one-liner that answers "what do you do" |
| 4 | Add a **"Impact"** subsection on About: bullet list with **quantified outcomes** (replace "$xxxM" placeholders with real ranges, e.g. "low 9-figures iRev impact"). Recruiters scan for numbers. | Removes guesswork |
| 5 | Add a **"Talks"** page — even 2-3 entries (internal Atlassian, Meta, conference) | Authority signal |
| 6 | Add a **"Publications"** page linking to your PhD papers + Google Scholar | You're a PhD — this is wasted credibility currently |
| 7 | Add a **Newsletter signup on home** (Follow.it form already exists; promote to homepage) | Email > follower count |
| 8 | Get a custom domain email like `abdullah@newabdullah.com` (proton, fastmail, ImprovMX) | Shows up in Resend/Mailgun signups, looks pro |

### Tier 2 (this month)

| # | Action | Why |
|---|---|---|
| 9 | Create a `/projects` page: 4-6 cards (RecSys series, LLM eval framework, agentic onboarding agent, etc.) with screenshot + repo link + one-liner | VCs/founders want to see ship cadence |
| 10 | Add **"What I'm building now"** page (Derek Sivers-style `/now`) — updated monthly | Signals you're alive and shipping |
| 11 | Add a hero homepage CTA: "**Hiring AI talent or building something? → reach out**" | Direct conversion path |
| 12 | Recruit a **photographer for headshot** (or hire ~$200 on Fiverr). Replace `aboutme.jpg` | Photo is your handshake |
| 13 | Set up **Google Search Console + Bing Webmaster Tools**, submit sitemap | See what people search to find you |
| 14 | Add **JSON-LD Person schema** on homepage | Google may show your name in side-knowledge-panel |

### Tier 3 (when you have a story to tell)

- Open-source one substantial repo (e.g. "minimal RL fine-tuning for RAG") — these go viral on HN
- Get one talk recorded and put on YouTube → embed on Talks page
- Write a "What I learned at Meta about Ads ML" retrospective (these always go viral)

---

## Part 4 — Recommendations to help learners (and grow a following)

The compounding asset is **a content engine that helps people**. VCs/recruiters notice it as a byproduct.

### Content strategy

**Pillars** (commit to 2-3, don't sprawl):

| Pillar | Audience | Cadence |
|---|---|---|
| **Applied RecSys / Ads ML deep dives** (your Meta experience) | Senior ML engineers | 1 long-form / month |
| **Agentic AI patterns** (your current work) | Practitioners building agents | 1 long-form / 6 weeks |
| **RL fine-tuning for LLMs** (your current edge) | LLM engineers / researchers | 1 long-form / 6 weeks |
| **Career notes** (Bangladesh → KSA → Qatar → FIU → Meta → Aisera → Atlassian) | Aspiring AI engineers | 1 / quarter — these go viral on LinkedIn |

### Post format template (steal from Lilian Weng + Eugene Yan)

```
1. TL;DR (3-5 bullets — answer "what will I learn?")
2. The problem (1-2 paragraphs — make me care)
3. Background / prerequisites (linked, not re-explained)
4. Core content (with diagrams, code, math where needed)
5. Practical takeaways (a checklist or table)
6. References / further reading
7. Companion GitHub repo link
```

### Concrete first post ideas (write these in order)

1. **"What's a real evaluation framework for RecSys? Lessons from Meta Ads"** (you've started the series; finish it as your authoritative reference)
2. **"How we built SMART Answer for Jira/Confluence: an honest agentic AI architecture"** (current work — get Atlassian PR review)
3. **"RL fine-tuning for retrieval-augmented LLMs: practitioner's guide"** (your edge — very few people writing this well)
4. **"From PhD to Meta to Atlassian: a non-American immigrant's path in applied AI"** (career — this WILL go viral if honest)
5. **"Why your RAG system feels dumb (and how RL-fine-tuned LLMs fix it)"** (current work — opinionated takes go viral)

### Discovery / SEO content

- **Tag pages** as landing pages (currently auto-generated by Hugo, good)
- **Internal linking**: every new post should link to ≥3 older posts
- **Comments**: enable Giscus (free, GitHub-based) — `comments: true` in frontmatter already, just needs `[params.giscus]` config

---

## Part 5 — Multi-platform publishing (the POSSE pipeline)

Goal: *"linked with LinkedIn, Substack, Medium and other sites so that once I publish something it will publish everywhere else."*

The right pattern is **POSSE** (Post Once, Syndicate Everywhere). Your site = canonical URL. Other platforms get copies with `rel=canonical` pointing back to your site so Google credits YOU.

### Recommended stack

| Platform | Strategy | Tool / Method |
|---|---|---|
| **Your site (`newabdullah.com`)** | Canonical home. Markdown source. | Hugo + GitHub Actions (already done) |
| **RSS feed** | Hub everything else pulls from | `https://www.newabdullah.com/index.xml` (already exists) |
| **Substack** | Has built-in **"Import from RSS"** feature. Set up once. Each post auto-mirrors. Canonical link is automatic. | Substack → Settings → Import → RSS |
| **Medium** | Has **"Import a story"** by URL. Honors `rel=canonical` (you set this on Medium per import). | Medium UI manual import, or `medium-cli` |
| **dev.to** | Built-in RSS import setting. `canonical_url` field per post. Tech audience. | dev.to → Settings → Extensions → "Publishing from RSS" |
| **Hashnode** | Same — RSS import + canonical. AI/ML community is strong here. | Hashnode → Settings → Import → RSS |
| **LinkedIn** | No native RSS import. Best ROI is a **manual hand-crafted post** with a different "hook" for the LinkedIn audience + link back to your blog. | Hand-write + schedule via Buffer / Taplio |
| **X (Twitter)** | Auto-tweet via Echofeed (free) or Buffer | Echofeed.app: RSS → X / Bluesky / Mastodon / Threads |
| **Bluesky / Mastodon / Threads** | Same auto-cross-post via Echofeed | Echofeed.app |

### Concrete plan (~2 hours of setup, then zero ongoing work)

**One-time setup:**

1. Create accounts: Substack, Medium, dev.to, Hashnode, Bluesky (if missing)
2. **Substack**: Settings → Imports → "Import an RSS feed" → paste `https://www.newabdullah.com/index.xml`
3. **dev.to**: Settings → Extensions → "Publishing from RSS" → paste feed URL → enable "use canonical URL"
4. **Hashnode**: Settings → Headless / Import → enable RSS import with canonical
5. **Echofeed.app** (free tier): connect feed → connect X account + Bluesky account → enable auto-post
6. **Medium**: NO auto-import worth it; do manual UI import per post (5 min each, sets canonical correctly)
7. **LinkedIn**: hand-write a custom version for each post. Use **Buffer** ($5/mo) or **Typefully** to schedule

**Per-post workflow after that:**

```
1. Write markdown locally
2. git push origin main           (dev site updates)
3. scripts/promote-to-prod.sh     (prod site updates → RSS feed updates)
4. AUTO: Substack, dev.to, Hashnode, X, Bluesky pull from RSS (within minutes)
5. MANUAL: Medium (5 min import) + LinkedIn (10 min custom post)
   → Both should link back to your blog as canonical
```

### Optional: automate LinkedIn too

LinkedIn's official API has UGC posting. Build a GitHub Action that on push, calls LinkedIn API. Setup cost: ~3 hours. Tools/libraries:

- `linkedin-api` (unofficial Python)
- Or: pay for **Typefully** / **Buffer** / **Taplio** (~$15-30/mo) — schedule from a queue
- Recommendation: don't auto-post to LinkedIn yet. LinkedIn rewards native posts with images and hooks; auto-posted RSS underperforms.

### What to NOT cross-post

- Don't paste your full blog content into LinkedIn posts (LinkedIn de-prioritizes links). Instead: write a custom 200-word LinkedIn post with the hook + image + "full article here →" link.
- Don't post the same headline everywhere. Customize for platform tone.

---

## Suggested execution order

If I were doing this for you, the order would be:

| Sprint | Focus | Outcome |
|---|---|---|
| **Week 1 (this weekend)** | SEO Tier 1 quick wins (#1-#8) | Site is professional in 5 sec |
| **Week 2** | Talks + Publications + Projects pages | Credibility surface area 2x |
| **Week 3** | Substack + dev.to + Hashnode + Echofeed setup | Distribution pipeline live |
| **Week 4** | Write first real post on the new pipeline | Validate end-to-end |
| **Ongoing** | 1 long-form post / month + 1 LinkedIn / week | Compound |

---

### What I can do for you right now (just say which)

I can immediately implement these without further input — just pick:

- **(A)** Fix all Tier 1 SEO gaps (meta description, og:image, Twitter cards, canonical, JSON-LD, slug fix, default social card image). ~1 hour of work, ready-to-deploy.
- **(B)** Scaffold the new pages: `/talks`, `/publications`, `/projects`, `/now`. Empty stubs you fill in.
- **(C)** Wire up Giscus comments (uses GitHub Discussions, free, no spam).
- **(D)** Generate a 1200×630 default social card image and wire it as fallback `og:image`.
- **(E)** Set up Google Search Console + sitemap submission via a verification file.
- **(F)** Write a follow.it / Substack signup widget for the homepage hero.
- **(G)** All of the above as one big PR.
