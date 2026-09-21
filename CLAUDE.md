# newabdullah.com — writing standard

Hugo blog. Author in this repo (`pwaabdullah.github.io`); `newabdullah.com` is a
force-pushed mirror, never edit it. Push to `main` is the whole deploy.

## What every post is for

Three goals, in this order. A section that serves none of them gets cut.

1. **Rapid-fire recall.** The reader can answer an interview question from memory.
2. **Plain-words understanding.** The reader can explain the concept out loud, to
   another human, in under thirty seconds, without writing anything down.
3. **Monday-morning decisions.** The reader knows what to actually do differently
   on their own project tomorrow.

Not a textbook chapter. Not a syllabus. An engineer's reference.

## The house style is `content/posts/ml-models-guide.md`

When unsure, open that post and match it. It leads with plain language, gives each
model the same three beats (what it is really doing → when to reach for it → where
it breaks), and carries almost no notation. That is the target.

## Rules

**1. Plain first. Never open a section with a formula.**
The first sentence after a heading must make sense to someone who does not know the
notation yet. State the idea in words, then show the math.

> Bad: `### 4.4 GELU` then immediately `$$\text{GELU}(x)=x\,\Phi(x)$$`
>
> Good: "ReLU makes a hard call: keep the value or zero it. GELU makes a soft one,
> scaling each input by how likely it is to matter. Then the formula."

**2. Keep the formula. Demote it.**
Formulas stay, they are part of the value. But they are the *receipt*, not the
argument. Every formula is preceded by what it means in words and followed by what
it changes for the reader. A formula that sits alone between two paragraphs is a
bug.

**3. Every concept needs one say-it-out-loud sentence.**
One sentence, no symbols, that the reader could repeat in an interview and sound
like they understand it. If you cannot write that sentence, you do not understand
the concept well enough to publish it yet.

**4. Every section lands on a decision.**
What to reach for, what to set it to, what breaks. A purely descriptive section
fails goal 3. The models post's "Reach for it when / It breaks when" is the pattern.

**5. Notation budget.**
Introduce a symbol only if it appears more than once. Otherwise write the word.
"the probability given to the correct class" beats \(p_t\) on first use.

**6. Numbers beat adjectives.**
"caps at 0.25, so the gradient shrinks at least fourfold per layer" beats "small
gradients". Where a claim can be measured, measure it and print the number.

**7. Q&A covers this post only.**
The rapid-fire section is an index into what this post actually taught. Never add a
question the post does not answer, however good the question is. It belongs in the
post that covers it.

**8. Define jargon inline, on first use, in a clause.**
"a matrix preconditioner (a transform that rescales the gradient using curvature
across a whole weight matrix, not one number per parameter)".

**9. The opening works cold.**
Most readers arrive from search, not from the previous post. Cross-link the series,
but never depend on it to be understood.

**10. Tone: casual and human, but tight.**
Write like you are explaining it to a colleague at a whiteboard, not presenting at a
conference. Use contractions. Address the reader as "you". Ask the question before
answering it ("Why did that matter so much?"). Short sentences are allowed to stand
alone. Prefer "it's still around because" over "it survives where X is the point",
and "you pay an exp for it" over "costs an exp on the negative branch".

Casual does not mean padded or jokey. Every sentence still earns its place, the
numbers stay, the formulas stay. No em dashes anywhere (site-wide convention); use
commas, colons or parentheses.

## Figures

Earn their place by proving something the text asserts. Simulate the result rather
than drawing the conclusion you want: fit the model, print the number, put it in the
caption. Hand-written SVG, white card, slate palette, matching
`static/posts/*/diagrams/`. Validate any categorical palette for colourblind
separation before using it. **Render to PNG and look at it** before publishing;
label collisions and axis clipping are invisible in source.

## Mechanics that bite

- **A raw `<` followed by a letter inside LaTeX destroys the equation.** Goldmark runs
  with `unsafe: true` and reads `<t` as an HTML tag. Write `\lt` and `\gt`. Prefer
  `\mid` over `|` for conditional bars. Lint before building:
  `grep -n '<[a-zA-Z]' content/posts/<file>.md` must return nothing.
- **One `<h1>` per page.** It comes from the frontmatter title. Never put `# ` in the body.
- **One post per calendar date.** Never two posts sharing a `date:`, never a future
  date (`buildFuture: false` makes a future-dated post silently vanish).
- **Raw working notes** (`content/posts/*-raw.txt`) are gitignored and excluded via
  `ignoreFiles`; anything else dropped in `content/` gets published.
- Slug changes must move `static/posts/<slug>/` to match, or every figure 404s.
