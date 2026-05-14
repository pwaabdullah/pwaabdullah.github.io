---
title: "Projects"
layout: "single"
url: "/projects/"
draft: true   # Hidden until real project list is curated — flip to false when ready
description: "Projects, open-source contributions, and writing by Abdullah Al Mamun on RecSys, agentic AI, and LLM fine-tuning."
summary: "Selected open-source work, blog series, and applied ML projects."
ShowToc: false
ShowReadingTime: false
ShowBreadCrumbs: false
ShowWordCount: false
hidemeta: true
---

<section class="page-section">
  <p class="page-lede">
    A mix of writing, open-source code, and applied ML projects. Updated periodically.
  </p>

  <div class="project-grid">

    <article class="project-card">
      <h3 class="project-title">The Evaluation of RecSys (blog series, 4 parts)</h3>
      <p class="project-summary">
        A deep-dive series tracing the evolution of recommendation systems from collaborative
        filtering and matrix factorization through Factorization Machines, neural collaborative
        filtering, Wide &amp; Deep, DeepFM, DIN, DLRM, and AdaTT, with practical evaluation
        strategies at each step.
      </p>
      <div class="project-tags">
        <span>RecSys</span><span>Deep Learning</span><span>Evaluation</span>
      </div>
      <div class="project-links">
        <a href="/posts/the-evaluation-of-recsys---part-1/">Part 1 →</a>
        <a href="/posts/the-evaluation-of-recsys-part2/">Part 2 →</a>
        <a href="/posts/the-evaluation-of-recsys-part-3/">Part 3 →</a>
      </div>
    </article>

    <!--
      Add new project cards using this template:

      <article class="project-card">
        <h3 class="project-title">Project name</h3>
        <p class="project-summary">1-3 sentence description of what it is and why it matters.</p>
        <div class="project-tags">
          <span>Tag 1</span><span>Tag 2</span>
        </div>
        <div class="project-links">
          <a href="LINK" target="_blank" rel="noopener">GitHub →</a>
          <a href="POST_LINK">Write-up →</a>
        </div>
      </article>
    -->

    <article class="project-card project-card-stub">
      <h3 class="project-title">More projects coming soon</h3>
      <p class="project-summary">
        Curating the list of open-source repos and applied ML projects to highlight here.
        Check back, or browse my <a href="https://github.com/pwaabdullah" target="_blank" rel="noopener">GitHub profile</a> in the meantime.
      </p>
    </article>

  </div>
</section>

<style>
.page-lede { font-size: 1.05rem; color: var(--secondary); margin-bottom: 2rem; }
.project-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
}
@media (min-width: 700px) {
  .project-grid { grid-template-columns: 1fr 1fr; }
}
.project-card {
  padding: 1.25rem 1.5rem;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--entry);
  display: flex;
  flex-direction: column;
}
.project-card-stub {
  border-style: dashed;
  opacity: 0.8;
}
.project-title {
  font-size: 1.1rem;
  margin: 0 0 0.5rem 0;
  line-height: 1.35;
}
.project-summary {
  font-size: 0.95rem;
  color: var(--content);
  margin: 0 0 0.8rem 0;
  line-height: 1.55;
  flex-grow: 1;
}
.project-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.8rem;
}
.project-tags span {
  font-size: 0.75rem;
  background: var(--tertiary);
  color: var(--secondary);
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
.project-links {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.9rem;
}
.project-links a {
  text-decoration: none;
  color: var(--primary);
  border-bottom: 1px dashed var(--secondary);
}
.project-links a:hover { border-bottom-style: solid; }
</style>
