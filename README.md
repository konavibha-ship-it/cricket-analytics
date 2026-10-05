# 🏏 Cricket Performance Analytics Project

## Overview
An end-to-end data analytics project built on ball-by-ball IPL cricket data 
(sourced from Cricsheet.org). The project covers the full analytics pipeline: 
raw data parsing, cleaning, SQL-based analysis, exploratory data analysis, 
statistical insights, a machine learning win-prediction model, and an 
interactive live dashboard.

## Skills Demonstrated
- **Data Engineering**: Parsed raw JSON match files into structured relational tables
- **Data Cleaning**: Handled missing values, inconsistent team names, duplicates (pandas)
- **SQL**: Built a SQLite database and wrote aggregate queries (GROUP BY, CASE, JOIN logic)
- **Exploratory Data Analysis**: Statistical summaries, correlation analysis
- **Data Visualization**: matplotlib, seaborn static charts + Streamlit interactive charts
- **Machine Learning**: Random Forest classifier for match outcome prediction
- **Dashboarding**: Built and deployed a live interactive Streamlit app
- **Version Control**: Git/GitHub for project tracking

## Key Findings
- Teams that won the toss also won the match **[X]%** of the time
- **[Player Name]** is the top run scorer across the dataset with **[X]** runs
- **[Player Name]** leads in wickets taken with **[X]** wickets
- The **[Powerplay/Middle/Death]** phase showed the highest scoring strike rate
- Baseline win-prediction model (pre-match features only) achieved **[X]%** accuracy

## Project Structure

📐 [System Architecture](docs/architecture.md) · 🔧 [Engineering Challenges](docs/engineering-challenges.md)