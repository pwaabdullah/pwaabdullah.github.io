---
title: "Publications"
layout: "single"
url: "/publications/"
description: "Selected peer-reviewed publications by Abdullah Al Mamun, PhD — deep learning and applied machine learning for computational biology, cancer biomarker discovery, and biomedical informatics."
summary: "Selected peer-reviewed publications on deep learning and applied ML for computational biology."
ShowToc: false
ShowReadingTime: false
ShowBreadCrumbs: false
ShowWordCount: false
hidemeta: true
---

<section class="page-section">

<p class="page-lede">
  A curated set of peer-reviewed work applying deep learning and machine learning
  to computational biology — cancer biomarker discovery, multi-cancer classification,
  and biomedical interpretability. Full list and live citation counts on
  <a href="https://scholar.google.com/citations?user=iMIqKjcAAAAJ&hl=en" target="_blank" rel="noopener">Google Scholar</a>.
</p>

<article class="pub-entry">
  <p class="pub-year">2021</p>
  <h3 class="pub-title">
    <a href="https://doi.org/10.3390/ijms222111919" target="_blank" rel="noopener">
      Multi-run concrete autoencoder to identify prognostic lncRNAs for 12 cancers
    </a>
  </h3>
  <p class="pub-authors">
    <strong>Abdullah Al Mamun</strong>, Raihanul Bari Tanvir, Masrur Sobhan, Kalai Mathee,
    Giri Narasimhan, Gregory E. Holt, Ananda Mohan Mondal
  </p>
  <p class="pub-venue">
    <em>International Journal of Molecular Sciences</em>, 22(21), 11919 &nbsp;·&nbsp; MDPI &nbsp;·&nbsp; Q1
  </p>
  <p class="pub-summary">
    Deep-learning approach (concrete autoencoder) for biomarker discovery across
    12 cancer types — finds prognostic long non-coding RNAs that survive across
    independent runs, giving clinicians a stable feature set.
  </p>
</article>

<article class="pub-entry">
  <p class="pub-year">2020</p>
  <h3 class="pub-title">
    <a href="https://doi.org/10.1109/BIBM49941.2020.9313332" target="_blank" rel="noopener">
      Pan-cancer feature selection and classification reveals important long non-coding RNAs
    </a>
  </h3>
  <p class="pub-authors">
    <strong>Abdullah Al Mamun</strong>, Wenrui Duan, Ananda Mohan Mondal
  </p>
  <p class="pub-venue">
    <em>IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2020)</em>
  </p>
  <p class="pub-summary">
    Machine-learning feature-selection pipeline for pan-cancer classification —
    identifies a compact set of lncRNAs that distinguish tumor types, useful for
    diagnostic ML models on TCGA.
  </p>
</article>

<article class="pub-entry">
  <p class="pub-year">2020</p>
  <h3 class="pub-title">
    <a href="https://doi.org/10.1109/BIBM49941.2020.9313450" target="_blank" rel="noopener">
      Deep learning to discover cancer glycome genes signifying the origins of cancer
    </a>
  </h3>
  <p class="pub-authors">
    <strong>Abdullah Al Mamun</strong>, Masrur Sobhan, Raihanul Bari Tanvir,
    Charles J. Dimitroff, Ananda Mohan Mondal
  </p>
  <p class="pub-venue">
    <em>IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2020)</em>
  </p>
  <p class="pub-summary">
    Deep-learning pipeline that surfaces cancer-origin glycome genes from
    high-dimensional expression data — bridging interpretable ML and tumor biology
    to point at tissue-of-origin signatures.
  </p>
</article>

<article class="pub-entry">
  <p class="pub-year">2020</p>
  <h3 class="pub-title">
    <a href="https://doi.org/10.1109/BIBM49941.2020.9313426" target="_blank" rel="noopener">
      Deep learning to discover genomic signatures for racial disparity in lung cancer
    </a>
  </h3>
  <p class="pub-authors">
    Masrur Sobhan, <strong>Abdullah Al Mamun</strong>, Raihanul Bari Tanvir,
    Maria J. Alfonso, Pia Valle, Ananda Mohan Mondal
  </p>
  <p class="pub-venue">
    <em>IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2020)</em>
  </p>
  <p class="pub-summary">
    Deep learning surfaces genomic signals that differ between racial groups in
    lung cancer — a societal-impact application of ML to a healthcare-equity
    problem.
  </p>
</article>

<article class="pub-entry">
  <p class="pub-year">2019</p>
  <h3 class="pub-title">
    <a href="https://doi.org/10.1145/3307339.3343249" target="_blank" rel="noopener">
      Long non-coding RNA based cancer classification using deep neural networks
    </a>
  </h3>
  <p class="pub-authors">
    <strong>Abdullah Al Mamun</strong>, Ananda Mohan Mondal
  </p>
  <p class="pub-venue">
    <em>Proceedings of the 10th ACM International Conference on Bioinformatics,
    Computational Biology and Health Informatics (ACM-BCB 2019)</em>
  </p>
  <p class="pub-summary">
    End-to-end deep neural network for multi-cancer classification using lncRNA
    expression — outperforms classical ML baselines on the TCGA pan-cancer dataset.
  </p>
</article>

<p class="pub-footnote">
  For the complete list, conference proceedings, and live citation counts, see
  <a href="https://scholar.google.com/citations?user=iMIqKjcAAAAJ&hl=en" target="_blank" rel="noopener">Google Scholar</a>.
</p>

</section>

<style>
.page-section { max-width: 880px; margin: 0 auto; }
.page-lede { font-size: 1.02rem; color: var(--secondary); margin-bottom: 2rem; line-height: 1.6; }
.pub-entry {
  padding: 1.4rem 0;
  border-bottom: 1px solid var(--border);
}
.pub-entry:last-of-type { border-bottom: none; }
.pub-year {
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  color: var(--brand);
  font-weight: 700;
  text-transform: uppercase;
  margin: 0 0 0.35rem 0;
}
.pub-title {
  font-size: 1.1rem;
  margin: 0 0 0.45rem 0;
  line-height: 1.4;
}
.pub-title a {
  color: var(--primary);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s ease, color 0.2s ease;
}
.pub-title a:hover {
  color: var(--brand);
  border-bottom-color: var(--brand);
}
.pub-authors {
  font-size: 0.93rem;
  color: var(--content);
  margin: 0.25rem 0;
  line-height: 1.5;
}
.pub-venue {
  font-size: 0.88rem;
  color: var(--secondary);
  margin: 0.25rem 0;
}
.pub-summary {
  font-size: 0.93rem;
  color: var(--content);
  margin: 0.55rem 0 0 0;
  line-height: 1.6;
}
.pub-footnote {
  margin-top: 2.5rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  font-size: 0.92rem;
  color: var(--secondary);
  text-align: center;
}
.pub-footnote a {
  color: var(--brand);
  text-decoration: none;
}
.pub-footnote a:hover { text-decoration: underline; }
</style>
