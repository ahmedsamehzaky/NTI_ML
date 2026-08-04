# Global AI Coding Instructions

## Main goal

Complete the requested task correctly while using the smallest relevant context, the fewest necessary tool calls, and the minimum number of code changes.

## Scope control

- Understand the exact task before inspecting files.
- Identify the smallest relevant scope.
- Do not scan the entire repository unless required.
- Search for relevant filenames, functions, classes, imports, and error messages before opening files.
- Read only files directly related to the current task.
- Do not repeat repository exploration already completed in the same session.
- Stop investigating once enough information exists to implement and validate the requested change.

## File reading

- Read only relevant sections of large files.
- Do not print complete datasets, notebooks, logs, lock files, or generated artifacts.
- Do not read binary model files as text.
- Ignore unrelated directories such as `.git`, `.venv`, `venv`, `__pycache__`, `node_modules`, `dist`, and `build`.
- Prefer targeted search over opening many files.

## Editing

- Modify the minimum number of files necessary.
- Do not refactor unrelated code.
- Do not rename files, folders, functions, classes, or variables unless required.
- Preserve the existing project structure.
- Preserve working behavior unrelated to the requested task.
- Do not add optional features after completing the task.
- Do not rewrite a complete file when a small edit is sufficient.
- Before editing multiple files, create a plan with no more than five steps.

## Commands and validation

- Use concise command output.
- Run the narrowest useful validation first.
- Prefer one relevant test over the full test suite.
- Prefer syntax checks, imports, or one target function before running the complete application.
- Do not rerun an entire notebook for a small change.
- Do not retrain unrelated machine-learning models.
- Do not install or upgrade unrelated dependencies.
- Check the active environment and `requirements.txt` before installing packages.

## Python and machine learning

When inspecting data, report only what is necessary:

- Shape.
- Column names.
- Data types.
- Missing-value counts.
- Duplicate count.
- Target distribution.
- A small sample only when required.

When modifying an ML workflow:

- Reuse existing preprocessing when valid.
- Preserve feature order.
- Reuse the current train-test split when valid.
- Reuse saved model artifacts when valid.
- Retrain only affected models.
- Generate only the metrics and plots needed for the task.
- Prevent data leakage.
- Fit preprocessing only on training data.

## Jupyter notebooks

- Locate the relevant cells before editing.
- Do not repeat completed EDA or preprocessing.
- Reuse existing variables and pipeline objects.
- Keep cell outputs concise.
- Do not print complete DataFrames.
- Avoid duplicating code across cells.

## Streamlit

- Inspect the main Streamlit file and directly imported local modules first.
- Check model loading separately before running the full interface.
- Preserve the current UI unless redesign is requested.
- Do not modify training code for a UI-only problem.
- Do not retrain models unless artifacts are missing or invalid.
- Check `requirements.txt` before installing packages.

## Safety and control

- Ask before deleting files or replacing large working sections.
- Do not expose secrets, API keys, tokens, or credentials.
- Do not modify environment files containing secrets.
- Do not push, publish, merge, or deploy unless explicitly requested.

## Response

At the end, report only:

1. What changed.
2. Files changed.
3. Validation performed.
4. Remaining issues.

Keep the response focused and do not recommend unrelated work.

---

# ML Engineer Instructions

For machine-learning tasks:

1. Identify the target, problem type, metrics, and current project stage.
2. Do not repeat valid EDA, cleaning, preprocessing, or training.
3. Split data before fitting imputers, scalers, encoders, feature selectors, or resampling methods.
4. Prevent target leakage and duplicate leakage.
5. Prefer reproducible `Pipeline` and `ColumnTransformer` workflows.
6. Begin with a suitable baseline before complex models.
7. Train only models relevant to the requested comparison.
8. Investigate unrealistic perfect metrics before accepting them.
9. Preserve feature names, order, preprocessing, and saved artifacts.
10. Validate inference with a realistic sample before Streamlit integration.
11. Do not retrain unrelated models.
12. Do not print complete datasets or notebook contents.
13. Make the minimum necessary changes.
14. Report changes, metrics, artifacts, validation, and limitations.

Do not scan the complete repository unless the task requires it.
