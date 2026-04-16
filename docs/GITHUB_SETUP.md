# Publishing this project to GitHub

Target repository: **[https://github.com/Ellison097/ecommerce-recommendation-study](https://github.com/Ellison097/ecommerce-recommendation-study)** (create this empty repo on GitHub first if it does not exist).

## What is included vs excluded

- **Included:** `src/`, `configs/`, `run_experiment.py`, `requirements.txt`, `README.md`, `data/ratings/` (cached CSV), and generated metrics/figures if you choose to commit them.
- **Excluded by `.gitignore`:** `generate_report.py`, `generate_outputs.py` (local report/poster helpers), and `results/**/*.pdf` / `results/**/*.pptx`.

Keep a **private backup** of `generate_report.py` on your machine; it will not be pushed.

## Steps

1. Log in to GitHub CLI (optional but convenient):

   ```bash
   gh auth login
   ```

2. Create the remote repository (if needed):

   ```bash
   gh repo create Ellison097/ecommerce-recommendation-study --public --source=. --remote=origin --push
   ```

   Or create an empty repo named `ecommerce-recommendation-study` in the GitHub web UI, then:

   ```bash
   git remote add origin https://github.com/Ellison097/ecommerce-recommendation-study.git
   git branch -M main
   git add .gitignore README.md requirements.txt configs/ src/ run_experiment.py data/ratings/
   git add results/metrics/ results/figures/   # optional: precomputed outputs
   git commit -m "Initial commit: e-commerce recommender benchmark code and data"
   git push -u origin main
   ```

3. If `generate_report.py` was previously tracked, stop tracking it while keeping the local file:

   ```bash
   git rm --cached generate_report.py 2>/dev/null || true
   ```

Ensure the repository stays **maintained** (README, tags, or releases) to satisfy the practicum requirement for a maintained repo, not a one-time upload.
