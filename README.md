
# newabdullah.com – Personal Tech Blog

Welcome to **newabdullah.com**, the personal website and blog of **Abdullah Al Mamun** – a Machine Learning Engineer passionate about cutting-edge advancements in **LLMs, Recommender Systems, Semantic Search, and Agentic AI**.

Hosted at 👉 https://newabdullah.com/  
Codebase 👉 [GitHub Repo](https://github.com/pwaabdullah/newabdullah.com/)

---

## 🧠 Tech Stack
- **Hugo** (Static Site Generator)
- **PaperMod Theme** (clean, responsive, and fast)
- **Markdown** for content
- **GitHub Pages** for hosting

---

## 🚀 Getting Started Locally

### Prerequisites
- [Install Hugo](https://gohugo.io/getting-started/install/)
- [Install Git](https://git-scm.com/)

### Clone and Run
```bash
git clone https://github.com/pwaabdullah/newabdullah.com.git
cd newabdullah.com
hugo server -D
```
Visit 👉 http://localhost:1313

---

## 📁 Project Structure
```
newabdullah.com/
├── content/           # Blog posts and pages
│   ├── posts/         # Blog articles
│   └── about/         # About me page (_index.md)
├── layouts/           # Custom layout overrides
├── themes/PaperMod/   # PaperMod theme (git submodule)
├── static/            # Static assets (images, css, etc.)
├── config.yml         # Site configuration (YAML-based)
└── README.md          # You're here 😄
```

---

## ✍️ Writing a New Blog Post
```bash
hugo new posts/my-post-title.md
```
Then edit `content/posts/my-post-title.md` with your content in Markdown.

---

## 🌍 Deployment (GitHub Pages)
1. The `public/` directory is built with:
   ```bash
   hugo --minify
   ```
2. Commit & push it to the appropriate branch if deploying manually.
3. Or use **GitHub Actions** to auto-deploy (setup recommended).

---

## 💡 Contribution
This is a personal project. However, if you spot a typo, feel free to open a PR or issue.

---

## 📬 Contact Me
- Blog: [https://newabdullah.com](https://newabdullah.com)
- LinkedIn: [linkedin.com/in/newabdullah](https://linkedin.com/in/newabdullah)
- Email: [aamcse@gmail.com](mailto:aamcse@gmail.com)

---

_Thank you for visiting! 🙏_

> Built with ❤️ using Hugo and PaperMod
