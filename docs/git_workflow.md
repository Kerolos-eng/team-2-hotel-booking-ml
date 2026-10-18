# Team 2 — GitHub collaboration and commit plan

The project description requires a 4–5 student team, and each student must contribute through their **own GitHub account**. “Team 2” is the team label, not a count of team members. Fill in the real roster and repository URL in `README.md` and `docs/progress_report_milestone1.md`.

The supplied ZIP was not itself a Git repository. Start the remote repository with the actual current files and then make real, reviewable follow-up contributions. Do not fabricate history, backdate commits, or set another student's author identity. A commit should reflect the person who actually made the change.

## 1. Repository owner: initialize and publish

Create an empty GitHub repository, then, from the extracted project folder:

```bash
cd hotel-project
git init
git add .
git status --short
git commit -m "chore: import Team 2 Milestone 1 project"
git branch -M main
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

Use the actual repository URL. Before pushing, check that `.gitignore` excludes `data/*.csv`, virtual environments, and generated fitted artifacts. Keep result tables, figures, source code, dependencies, and the report.

## 2. Every student: configure their own account and clone

Each contributor runs these commands using their own GitHub username and the name/email connected to their account. Configuration is local to this clone:

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
git config user.name "<your real name>"
git config user.email "<your verified GitHub email or GitHub noreply address>"
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux or `.venv\\Scripts\\Activate.ps1` in Windows PowerShell), then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 3. Branch, commit, review, and merge

Each student should take ownership of a real task, make incremental changes, and create a pull request for another member to review:

```bash
git checkout -b feat/<topic>
# make and test your own changes
git add <files-you-changed>
git commit -m "feat(<area>): <specific change>"
git push -u origin feat/<topic>
```

Open a pull request, describe the change and test command/output, request a review, and merge only after review. Use additional commits when they represent separate meaningful work. Do not use another member's Git credentials or identity.

## 4. Suggested division and commit-message examples

These are suggestions, not evidence of completed work. Reassign or combine roles to match the actual 4–5 students. Keep a short note of the real PR/commit URLs for the progress report.

| Suggested member/role | Example branch | Example messages for work actually completed |
|---|---|---|
| A — data, EDA, cleaning | `feat/data-eda` | `feat(data): add H2 shape and source validation`; `feat(eda): profile all source columns`; `feat(clean): document duplicate and invalid-row policy` |
| B — regression | `feat/regression-baseline` | `feat(reg): add leak-aware ADR feature set`; `feat(reg): report Dummy and Ridge cross-validation metrics`; `test(reg): verify pipeline feature selection` |
| C — classification | `feat/classification-baseline` | `feat(clf): add booking-time cancellation baseline`; `feat(clf): add labelled leakage ablation`; `docs(clf): explain prediction point and metrics` |
| D — clustering | `feat/clustering-baseline` | `feat(clu): add K-Means k scan with internal metrics`; `feat(clu): add cluster profile and PCA plot`; `docs(clu): document unsupervised feature choices` |
| E — CLI, tests, report (only if team has 5 students; otherwise split duties) | `feat/cli-report` | `feat(cli): add reproducible analyze clean and split commands`; `test: cover cleaning and leakage rules`; `docs: add Milestone 1 report and run instructions` |

Use truthful prefixes and scopes (`feat`, `fix`, `test`, `docs`, `chore`). Avoid committing the entire project under multiple members merely to make each person appear in the history. Later Milestone 2 tasks should also be split into genuine, reviewable contributions.

## 5. Final checks before the Milestone 1 ZIP

```bash
git status --short
git log --oneline --decorate -n 20
git log --oneline --author="<member name>"
git rev-parse HEAD
```

Verify that each named student's history is real, the project runs from the README commands, the ZIP contains the report and result tables/figures, and it excludes raw/generated CSVs, `.venv/`, and serialized fitted model artifacts. Paste the real repository URL and final submitted SHA into the README.
