---
title: "Resume"
layout: "single"
url: "/resume/"
description: "Resume of Abdullah Al Mamun, PhD — Sr. ML Engineer at Atlassian. Experience at Meta, Aisera, FIU; expertise in RecSys, agentic AI, LLM fine-tuning."
summary: "Resume page — HTML-readable summary plus downloadable PDF."
ShowToc: false
ShowReadingTime: false
ShowBreadCrumbs: false
ShowWordCount: false
hidemeta: true
---

<section class="page-section">
  <div class="resume-actions">
    <a class="btn-primary" href="/Abdullah_Resume.pdf" target="_blank" rel="noopener">
      <i class="fas fa-file-pdf"></i> Download PDF
    </a>
    <a class="btn-secondary" href="https://www.linkedin.com/in/newabdullah/" target="_blank" rel="noopener">
      <i class="fab fa-linkedin"></i> View on LinkedIn
    </a>
  </div>

  <p class="page-lede">
    Below is a quick HTML summary. For the formatted version with full project details, download the PDF above.
    A more detailed narrative is on the <a href="/about/">About page</a>.
  </p>

  <h2>Currently</h2>
  <p>
    <strong>Sr. ML Engineer · Atlassian · Central AI team</strong><br/>
    Building SMART Answer generation for Jira and Confluence Search using RL-fine-tuned LLMs
    and multi-Agent AI architecture.
  </p>

  <h2>Experience</h2>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Atlassian</strong> · Sr. ML Engineer, Central AI · 2025–Present</div>
    <p>
      SMART Answer generation for Jira and Confluence Search. RL-fine-tuned LLMs,
      multi-Agent AI architecture, retrieval-augmented generation, evaluation pipelines.
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Aisera</strong> · Senior Research Scientist · 2024–2025</div>
    <p>
      Led the design and deployment of agentic AI systems for IT/HR automation. Spearheaded the
      migration from GPT-4 to in-house fine-tuned LLaMA-3 models, significantly reducing inference
      cost. Improved real-time semantic search performance by 93% via RAG-pipeline optimizations.
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Meta</strong> · Research Scientist / ML Engineer · 2022–2024</div>
    <p>
      Ads ranking, personalization, and generative ad creation at scale. Multi-task multi-label
      models, Transformer-based sequence models, integration of LLMs into ad-creation workflows
      (CTR/CVR improvements). Contributed to AutoCA audience clustering.
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Florida International University</strong> · PhD researcher · 2017–2022</div>
    <p>
      PhD in CS — interpretable deep learning for early-stage cancer detection and drug
      recommendation. Multiple publications at ACM BCB, IEEE BIBM, and other venues.
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Qatar University</strong> · ML Researcher · 2017</div>
    <p>
      Won 2nd place in the GCC Robotics Challenge.
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>King Fahd University of Petroleum &amp; Minerals (KFUPM)</strong> · MS in Computer Engineering</div>
    <p>
      LSTM-based sentiment analysis on customer feedback (98% accuracy).
    </p>
  </article>

  <article class="exp-entry">
    <div class="exp-meta"><strong>Dhaka University of Engineering and Technology (DUET)</strong> · BS in CSE</div>
    <p>
      Foundation in computer science and early work in ML and NLP.
    </p>
  </article>

  <h2>Skills</h2>
  <div class="skills-grid">
    <div class="skill-group">
      <h4>LLMs &amp; Agents</h4>
      <p>LLM fine-tuning · RLHF / RLAIF · multi-Agent architectures · RAG · vector DBs · LangChain</p>
    </div>
    <div class="skill-group">
      <h4>RecSys &amp; Ranking</h4>
      <p>Collaborative filtering · MTML · DeepFM · DIN · DLRM · sequence models · A/B testing</p>
    </div>
    <div class="skill-group">
      <h4>Stack</h4>
      <p>PyTorch · Hugging Face · ONNX · Spark · Databricks · GCP · AWS · Docker · K8s</p>
    </div>
  </div>

  <p class="resume-footer">
    For the latest detailed version, see the <a href="/Abdullah_Resume.pdf" target="_blank" rel="noopener">PDF</a>
    or <a href="/contact/">reach out</a>.
  </p>
</section>

<style>
.resume-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 1rem 0 2rem 0;
}
.btn-primary, .btn-secondary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.1rem;
  border-radius: 999px;
  font-size: 0.95rem;
  text-decoration: none;
  font-weight: 600;
  transition: transform 0.15s ease;
}
.btn-primary {
  background: #00BFFF;
  color: #0F172A;
}
.btn-secondary {
  background: #0077B5;
  color: #fff;
}
.btn-primary:hover, .btn-secondary:hover { transform: translateY(-2px); }

.page-lede { font-size: 1rem; color: var(--secondary); margin-bottom: 2rem; }

.exp-entry {
  padding: 0.9rem 0;
  border-bottom: 1px solid var(--border);
}
.exp-entry:last-child { border-bottom: none; }
.exp-meta {
  font-size: 0.9rem;
  color: var(--secondary);
  margin-bottom: 0.4rem;
}
.exp-meta strong { color: var(--primary); }
.exp-entry p {
  margin: 0;
  line-height: 1.6;
  font-size: 0.97rem;
}

.skills-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  margin-top: 1rem;
}
@media (min-width: 700px) {
  .skills-grid { grid-template-columns: 1fr 1fr 1fr; }
}
.skill-group {
  padding: 1rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--entry);
}
.skill-group h4 {
  margin: 0 0 0.4rem 0;
  font-size: 0.95rem;
  color: #00BFFF;
}
.skill-group p {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.5;
}

.resume-footer {
  margin-top: 2rem;
  font-size: 0.9rem;
  color: var(--secondary);
}
</style>
