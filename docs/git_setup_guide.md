# Git and GitHub Submission Setup Guide

This guide explains how to initialize your Git repository, commit your code, create a GitHub repository, and share your project materials with your faculty coordinator.

---

## 1. Local Git Initialization

Open your terminal or VS Code PowerShell in this project folder (`c:\Users\yashr\Downloads\cca-3 cv`):

```bash
# Check if git is already initialized
git status

# If not initialized, initialize a new Git repository:
git init -b main

# Stage all project files (requirements, src, report, docs, tests, main.py)
git add .

# Create the initial commit
git commit -m "feat: complete CCA 3 Computer Vision assignment implementation (Units 3, 4, 5)"
```

> **Note on Large Media & Weights:**
> The `.gitignore` file is pre-configured to automatically exclude `*.mp4`, `*.avi`, `*.pt`, and virtual environments. This ensures your repository remains lightweight and within GitHub's 100 MB file limit.

---

## 2. Creating a Repository on GitHub

1. Log into your GitHub account at [https://github.com](https://github.com).
2. Click the **+** icon in the top-right corner and select **New repository**.
3. Repository name: `CCA3_Computer_Vision`
4. Description: `Practical Coding Assignment for CCA 3 Computer Vision (Units 3, 4, and 5) - MCA MIT-WPU`
5. Visibility: Select **Public** (or **Private** with faculty access).
6. **Do NOT check** "Add a README file" or ".gitignore" (these are already built in your project).
7. Click **Create repository**.

---

## 3. Pushing Local Code to GitHub

In your local terminal, run the following commands (replace `YOUR_GITHUB_USERNAME` with your real GitHub handle):

```bash
# Link your local repo to GitHub
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/CCA3_Computer_Vision.git

# Set default branch to main and push
git branch -M main
git push -u origin main
```

---

## 4. Google Drive Sharing for Large Video Files

Per assignment requirements:
> *"Large raw videos should be uploaded to Google Drive rather than unnecessarily committed to Git."*

1. Go to your Google Drive ([drive.google.com](https://drive.google.com)).
2. Create a folder named: `CCA3_CV_Submission_[YOUR_NAME]`.
3. Upload your original files:
   - `original_video.mp4` (15–30s self-recorded video)
   - `original_photo.jpg`
   - `second_photo.jpg`
   - Your 5–8 minute Project Explanation Video (`presentation_video.mp4`)
4. Right-click the folder $\to$ **Share**.
5. Add the faculty email addresses:
   - `rahul.inchal.inchal24@gmail.com`
   - `rahul.inchal.vf@mitwpu.edu.in`
6. Set permission to **Viewer** or **Editor**.
7. Copy the shareable link.
8. Paste the link into:
   - `report/academic_report.md`
   - `README.md`
   - Your final submission portal

---

## 5. Verification Checklist Before Submission

- [ ] All code runs cleanly via `python main.py`.
- [ ] Output figures exist in `outputs/`.
- [ ] `report/academic_report.md` has student name, roll number, and URLs filled in.
- [ ] Compiled PDF report generated at `report/CCA3_CV_Academic_Report.pdf`.
- [ ] Code successfully pushed to GitHub and link is accessible.
- [ ] Google Drive link shared with required faculty email addresses.
