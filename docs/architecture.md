# System Architecture

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        CS["Cricsheet JSON<br/>15 competitions, 2001-2026"]
        CB["Cricbuzz API<br/>live, on-demand"]
    end

    subgraph Pipeline["Data Pipeline"]
        CONV["convert_all_formats.py<br/>JSON → unified schema"]
        CLEAN["clean_all_formats.py<br/>team names, dates, dedup"]
        PARQ["Parquet files<br/>deliveries_clean.parquet<br/>matches_clean.parquet"]
        SQL["SQLite DB<br/>(local/notebook use)"]
        LOADER["data_loader.py<br/>cached, format-validated access"]
    end

    subgraph Tests["Validation"]
        PYTEST["pytest suite<br/>data quality + analytics logic<br/>23 tests"]
    end

    subgraph Analysis["Analysis & ML (src/)"]
        STATS["Hypothesis testing<br/>EDA"]
        WINPROB["Win probability models<br/>T20 + ODI, RandomForest"]
        EVAL["Model comparison<br/>SHAP, calibration, backtesting"]
        CLUSTER["K-Means clustering<br/>player archetypes"]
        ADV["Form rating, xRuns/xWickets,<br/>venue intelligence, par score,<br/>captaincy auditor, matchups"]
        FANTASY["Fantasy XI optimizer<br/>PuLP / linear programming"]
    end

    subgraph Dashboard["Streamlit Dashboard"]
        CORE["9 core tabs<br/>uses Parquet data only"]
        TRAJ["🏏 Trajectory popup<br/>physics sim, no dataset access"]
        LIVE["📡 Live lookup popup<br/>Cricbuzz API, no dataset access"]
    end

    subgraph Deploy["Deployment"]
        DOCKER["Dockerfile"]
        CI["GitHub Actions CI"]
        CLOUD["Streamlit Cloud"]
    end

    CS --> CONV --> CLEAN --> PARQ
    PARQ --> SQL
    PARQ --> LOADER
    LOADER --> Analysis
    PARQ -.validated by.-> PYTEST
    LOADER -.validated by.-> PYTEST

    Analysis --> CORE
    PARQ --> CORE
    CORE --> Dashboard
    TRAJ -.isolated.-> Dashboard
    LIVE -.isolated.-> Dashboard
    CB --> LIVE

    Dashboard --> DOCKER
    Dashboard --> CLOUD
    CI -.tests on push.-> PYTEST
```

**Design notes:**
- The two live add-ons (trajectory simulator, Cricbuzz lookup) are deliberately isolated — they import nothing from the core pipeline and the core pipeline never imports them, so either can fail or be removed without affecting the other.
- `data_loader.py` is the single point of entry for the dataset, added after an earlier bug where a script silently loaded all 15 competitions instead of IPL-only, caused by duplicated, inconsistent loading code across files.
- Models and large intermediate files are Parquet-compressed and gitignored where they exceed GitHub's 100MB limit; only the dashboard-required artifacts are committed.