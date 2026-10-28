# AI-Based Cloud Detection and Localization in Satellite Images

Software Project Management (Course 458) — University of Karachi
Instructor: Dr. Humera Azam | Group 4
---

# Local Setup Guide

Follow these instructions to set up and run the U-Net Satellite Patch Segmentation app on your local machine.

---

## 1. Prerequisites & Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd spm
   Create a Virtual Environment:

Bash
python3 -m venv .venv
Activate the Virtual Environment:

macOS / Linux:

Bash
source .venv/bin/activate
Windows (Command Prompt / Git Bash):

DOS
.venv\Scripts\activate
Windows (PowerShell):

PowerShell
.venv\Scripts\Activate.ps1
Install Dependencies:

Bash
pip install --upgrade pip
pip install streamlit tensorflow numpy pillow matplotlib pandas
File & Directory Requirements
Ensure your workspace contains the required sample patches and trained model weights before running:

Plaintext
spm/
├── interface/
│   ├── app.py
│   ├── segmentation_metrics.py
│   └── samples/                    <-- Ensure 12 .npz sample patches are here
│       ├── patch_105_6_by_5...npz
│       └── ...
└── model/
    ├── full_best.keras             <-- Place U-Net model checkpoint here
    └── band_stats.json             <-- Place normalization statistics here
3. Running the Streamlit App
Launch the application using Streamlit:

Bash
streamlit run interface/app.py --server.fileWatcherType none
The interface will automatically open in your default browser at http://localhost:8501.

## Team

| Name | Role |
|---|---|
| Tayyaba | Project Manager / QA Lead |
| Ayesha Mushtaq | Dataset & Preprocessing Lead |
| Fatima Fahad | AI Model Development Lead |
| Meerab Shafiq | Cloud Localization & Evaluation Lead |
| Samra Syed | Interface & Integration Lead |

## What this project does

Detects and localizes clouds in Landsat 8 satellite images using a
4-channel (Red, Green, Blue, NIR) U-Net segmentation model, trained on
the 38-Cloud dataset. Includes a Streamlit demo interface.

## Repo structure

```
cloud-detection-project/
├── data/
│   └── dataset_split/          # CSVs + split manifest (NOT the actual images)
├── model/
│   ├── unet_cloud_segmentation_skeleton.py
│   └── data_pipeline_cloud_segmentation.py
├── evaluation/
│   ├── segmentation_metrics.py
│   └── test_segmentation_metrics.py
├── interface/
│   ├── app.py
│   └── modules/
│       ├── preprocessing.py
│       ├── prediction.py
│       └── visualization.py
├── docs/
│   ├── project_charter.docx
│   ├── SOW.pdf
│   └── phase1_report.docx
├── .gitignore
└── README.md
```

## Important — what NOT to commit

- **The actual 38-Cloud image files (`.TIF` bands).** These are several
  GB and redownloadable from Kaggle — only the CSVs (file paths +
  labels) belong in the repo, not the images themselves.
- **`kaggle.json`** — this is your personal Kaggle API credential.
  Never commit it. It's already excluded in `.gitignore`, but double
  check before every push.
- Any large trained model checkpoint files (`.h5`, `.keras`, `.pt`) —
  these can get large fast; if we need to share a trained model later,
  we'll use Google Drive, not Git.

## Setup (for each team member)

1. Clone the repo:
   ```
   git clone https://github.com/<your-username>/cloud-detection-project.git
   cd cloud-detection-project
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```
3. Install dependencies for your component (each folder may need
   different packages — e.g. TensorFlow for `model/`, Streamlit for
   `interface/`).
4. Pull the dataset split CSVs from `data/dataset_split/` — the actual
   images still need to be downloaded separately via the Kaggle
   notebook (see `data/dataset_split/handoff_summary.md` if present).

## Workflow

- Create a branch for your work: `git checkout -b <your-name>/<short-task-name>`
- Commit small, working changes with clear messages
- Push and open a pull request before merging into `main`
- Tag a teammate for review if your change affects their component
  (e.g. changes to `model/` that affect `interface/`)


