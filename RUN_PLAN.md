# Multilingual RTL Experiment Run Plan
**Research:** Replication of arXiv:2512.14982 for RTL / Low-Resource Pakistani Languages  
**Updated:** 2026-09-21  

---

## 1. Status Legend
- [x] Done
- [~] In Progress
- [ ] Not Started
- [!] Blocked / Attention Required

---

## 2. Infrastructure & Environment Status
- [x] Install dependencies: `pandas`, `scipy`, `matplotlib`, `openpyxl`, `litellm`, `rapidfuzz`, `pymupdf`, `pypdf`
- [x] Persistent context memory created (`CONTEXT_MEMORY.md`, `AGENTS.md`)
- [x] Antigravity lifecycle hooks created (`.agents/hooks.json`)
- [x] GitHub MCP server configured (`.agents/mcp_config.json`)
- [~] GitHub CLI authentication (`gh auth login`) via one-time device code: **`F88B-2EB0`** at https://github.com/login/device
- [x] Dataset loaders verified: Punjabi (1,553 items) and Urdu (6,825 items)

---

## 3. Experiment Execution Roadmap

### Phase 1: Punjabi (`pa`) Full Run
- **Command:**
  ```powershell
  python run_punjabi.py --language pa --models gpt-4o-mini
  ```
- **Scope:** 7 tasks, 1,553 items × 5 repetition methods
- **Output:** `results_pa_YYYYMMDD_HHMMSS.json` + `.csv`

### Phase 2: Urdu (`ur`) Full Run
- **Command:**
  ```powershell
  python run_punjabi.py --language ur --models gpt-4o-mini
  ```
- **Scope:** 7 tasks, 6,825 items × 5 repetition methods
- **Output:** `results_ur_YYYYMMDD_HHMMSS.json` + `.csv`

### Phase 3: Pashto (`ps`) & Balochi (`bal`) Execution
- **Command:**
  ```powershell
  python run_punjabi.py --language ps --models gpt-4o-mini
  python run_punjabi.py --language bal --models gpt-4o-mini
  ```

### Phase 4: Analysis & Figure 1 Generation
- **Automated Execution:** Runs automatically after experiment completion.
- **Manual Execution:**
  ```powershell
  python analysis.py --csv results_pa_YYYYMMDD_HHMMSS.csv
  ```
- **Artifacts:**
  - `accuracy_table.csv`
  - `mcnemar_results.csv`
  - `figure1.png`
