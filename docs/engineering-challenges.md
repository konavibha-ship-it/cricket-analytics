# Engineering Challenges & Decisions

This project hit several real problems while scaling from a single-competition
(IPL) prototype to a 15-competition, 8.3M-row platform. Documenting them here
because the debugging process is arguably more representative of real data
engineering work than any single clean result.

## 1. GitHub's 100MB file limit (hit three separate times)

**What happened:** As the dataset grew — first with a 270MB tuned win-probability
model, then a 507MB combined CSV, then a 525MB SQLite database, then a 105MB
ODI match-state file — every one of these blew past GitHub's per-file limit.

**What I learned the hard way:** `.gitignore` only prevents *new* files from
being tracked. It does nothing for a file already committed in an earlier
history — `git add .` keeps re-staging it regardless. Fixing this required
`git reset --soft origin/main`, explicitly `git rm --cached` on the offending
file, and re-committing as a single clean commit with no trace of the large
blob anywhere in the pushed history.

**The actual fix:** converting flat CSVs to Parquet with Snappy compression —
columnar storage + dictionary encoding on repeated strings (team names,
dismissal types) shrank the 507MB deliveries file to well under 100MB with
no loss of data.

## 2. SQLite insert failures at scale

Loading 8.3M rows via `pandas.to_sql(..., method='multi')` hit SQLite's
~32,766 parameter-per-statement limit (chunksize × columns exceeded it).
Removing `method='multi'` switched to row-at-a-time parameterized inserts,
which is slower but has no such ceiling.

## 3. A silent file-save failure

One script (`win_probability_model.py`) stopped saving edits through VS Code
for reasons that were never fully identified — repeated edits, verified via
file timestamps, simply weren't landing on disk. The fix was deleting and
recreating the file from scratch rather than continuing to debug an
unreproducible editor/filesystem issue. Lesson: when a save *should* have
worked but a file's content and timestamp disagree, verify both directly
rather than trusting the editor's UI state.

## 4. A real bug caught by writing tests after the fact

An earlier version of the "milestone pace tracker" (predicting whether a
batter reaches 50/100 from their current state) compared a historical
innings' *final* total against the milestone, rather than its total *at the
matching checkpoint* — making it mathematically impossible to show a batter
progressing toward a milestone higher than their checkpoint. This produced a
silent, plausible-looking 0% for every query. Fixed by rebuilding the
function around cumulative ball-by-ball state. A regression test now locks
this behavior in (`test_milestone_tracker_uses_checkpoint_not_final_total`).

## 5. Model calibration: a negative result, kept honest

Backtesting the T20 win-probability model with calibration curves showed it
was overconfident above ~70% predicted probability (predicting 80%, actual
rate closer to 97%). I tried correcting this with `CalibratedClassifierCV`
using both isotonic and Platt (sigmoid) scaling. **Neither improved the
Brier score** — both made it marginally worse, and isotonic calibration also
ballooned the model file back to 270MB. Rather than force a fix that didn't
work, the original uncalibrated model was kept, and the limitation is
documented rather than hidden.

## 6. A format-scoping bug after expanding to 15 competitions

After adding Test/ODI/domestic-league data, `win_probability_model.py` kept
building ball-by-ball states from *all* formats combined (6.9M rows) instead
of IPL only, despite an intended filter — triggering a 30-minute run and then
a memory crash from the unfiltered RandomForest training. Root cause was the
duplicated, per-script data-loading pattern: every script had its own
`pd.read_csv(...)` call with its own (sometimes missing) filters. This is
what motivated building `src/data_loader.py` as a single, tested, cached,
format-validated loading layer that every script now shares.

## Takeaways

- Large-file git history needs managing deliberately, not just `.gitignore`.
- A script that "should" produce a result but produces something obviously
  wrong is worth more scrutiny than a script that crashes outright — crashes
  are loud, wrong-but-plausible numbers are silent.
- Not every optimization attempt (calibration, here) succeeds, and reporting
  that honestly is more valuable than hiding it or forcing a marginal "win."
- Centralizing data access after the fact is a legitimate, common refactor —
  the first version doesn't need to get this right, but recognizing *when*
  duplication becomes a real liability is a skill in itself.