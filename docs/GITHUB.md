# Publishing to GitHub

## First-time setup

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name: `neuro-edge-reram-simulator` (or your choice)
   - Do **not** initialize with README (you already have one)
   - Create repository

2. **Initialize and push from local**

   From the project root:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Neuro-Edge ReRAM Simulator"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/neuro-edge-reram-simulator.git
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your GitHub username (or org).

3. **Update README badge and links**
   - In README.md and CONTRIBUTING.md, replace `your-username` with your GitHub username in:
     - Badge URL
     - Clone URL
     - Repository / Documentation URLs
   - In `pyproject.toml`, update `[project.urls]` with your repo URL.

## Later updates

```bash
git add .
git commit -m "Your message"
git push
```

## CI

After the first push, GitHub Actions will run on every push and PR (see `.github/workflows/ci.yml`). Fix any failing tests before merging.
