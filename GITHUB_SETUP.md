# GitHub Setup & Push Guide — Glaucoma Detection Project

Follow this step-by-step guide to push this complete, reproducible project to GitHub and share it with your team.

---

## Step 1: Open the Project Directory

Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd C:\Users\TRIPU\.gemini\antigravity\scratch\glaucoma_detection
```

---

## Step 2: Inspect Git Status

Verify that the local Git repository is initialized and `.gitignore` is active:

```bash
git status
```

You will see untracked files like `README.md`, `PROJECT_STATUS.md`, `config/paths.example.yaml`, `src/`, `metadata/`, `splits/`, `results/`, `requirements.txt`, `environment.yml`.

> [!NOTE]
> The large image directories (`data/processed/`) and your local personal configuration (`config/paths.yaml`) are automatically excluded by `.gitignore`.

---

## Step 3: Stage All Tracked Files

Stage all project files for commit:

```bash
git add .
```

Re-check status to verify staged files:

```bash
git status
```

---

## Step 4: Create the Initial Commit

Create a local commit capturing the complete data preparation pipeline:

```bash
git commit -m "Initial commit: Glaucoma detection data prep pipeline, 5-fold CV, manifest, QC, and EDA"
```

---

## Step 5: Create a New GitHub Repository

1. Go to [GitHub](https://github.com/) and log in.
2. Click the **`+`** icon in the top right and select **New repository**.
3. Name your repository: `glaucoma_detection` (or your preferred name).
4. Set visibility to **Private** or **Public**.
5. **Do NOT** check "Initialize this repository with a README" (we already have a complete `README.md`).
6. Click **Create repository**.

---

## Step 6: Add the GitHub Remote & Push Code

Copy the repository URL from GitHub (e.g., `https://github.com/<your-username>/glaucoma_detection.git`) and run:

```bash
# Rename default branch to main if needed
git branch -M main

# Add remote origin URL
git remote add origin https://github.com/<your-username>/glaucoma_detection.git

# Push your code to GitHub
git push -u origin main
```

---

## Step 7: Add Teammates as Collaborators

1. In your GitHub repository, click **Settings**.
2. In the left sidebar, click **Collaborators**.
3. Click **Add people**.
4. Enter your teammate's GitHub username or email address and send an invite.

---

## Step 8: How Teammates Clone and Run

Your teammates can clone and set up their environment with:

```bash
git clone https://github.com/<your-username>/glaucoma_detection.git
cd glaucoma_detection

# Create local path config from template
cp config/paths.example.yaml config/paths.yaml
# (Edit config/paths.yaml to point to local raw dataset directories)

# Install requirements
pip install -r requirements.txt
```
