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

---

# ML Notebook Style

## Core Rule

The notebook is the main implementation.

For notebook-based ML projects, do all
training and analysis inside the notebook.

This includes:

- data loading
- data cleaning
- preprocessing
- feature engineering
- train/test split
- model creation
- model training
- evaluation
- metrics
- feature importance
- visualizations
- interpretation
- model saving
- report generation

Do not hide the real ML workflow inside
external Python scripts.

## Do Not Create by Default

Do not create:

- train.py
- training.py
- train_model.py
- run_training.py
- external training pipelines
- experiment runner scripts
- metric generator scripts

unless explicitly requested.

The notebook must not call an external
training script.

## Preferred Structure

Use:

project_name/
|
|-- app/
|-- data/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|-- .gitignore
|-- README.md
`-- requirements.txt

Do not create empty folders unnecessarily.

Preserve an existing project structure
before inventing a new one.

## src Rule

src/ is optional.

Use it only for genuinely reusable helpers
such as:

- prediction helpers
- input validation
- app utilities
- shared preprocessing helpers

Do not move the main training workflow
into src/.

## Naming Style

Prefer lowercase snake_case.

Good:

df_clean
df_model
target
identifier_columns
feature_names
data_path
model_path

Avoid unnecessary uppercase variables.

RANDOM_STATE is an acceptable exception.

ML notation such as X, X_train, X_test,
y_train and y_test is also acceptable.

## Code Layout

Code should look clean and readable.

For normal notebook code cells:

- target maximum 7 lines
- target about 40 characters per line
- use sensible exceptions when required

Visualization cells are exempt.

Do not compress code into ugly one-liners.

Use blank lines between logical blocks.

Prefer smaller cells instead of one giant
cell.

## Comments

Use short useful comments.

Examples:

# Remove identifier columns

# Check missing values

# Split the dataset

# Scale numerical features

# Plot the confusion matrix

Do not comment every obvious line.

## Markdown

The notebook must start with Markdown.

Explain:

- project objective
- dataset
- target
- problem type
- workflow
- preprocessing
- model
- evaluation
- interpretability
- deployment

Split the notebook into clear phases.

Every normal code cell should have useful
Markdown before it explaining what the
step does and why it is needed.

Avoid generic text such as:

"Now we run the next code."

## Typical Workflow

Use a logical flow such as:

Phase 1 - Business / Problem Understanding

Phase 2 - Environment Setup and Data Loading

Phase 3 - Data Understanding

Phase 4 - Data Quality Assessment

Phase 5 - Data Cleaning

Phase 6 - EDA

Phase 7 - Feature Engineering

Phase 8 - Data Preparation

Phase 9 - Model Building

Phase 10 - Training

Phase 11 - Evaluation

Phase 12 - Interpretability

Phase 13 - Model Saving

Phase 14 - Artifact Validation

Phase 15 - Final Summary

Adapt phases to the project.

## Data Understanding

Normally inspect:

- shape
- head
- columns
- data types
- missing values
- duplicates
- descriptive statistics
- target distribution

Do not create unnecessary analysis.

## Data Cleaning

Do not drop data blindly.

Explain important cleaning decisions.

Check:

- duplicates
- missing values
- invalid values
- data types
- identifier columns
- constant columns
- redundant features
- leakage risks

## Data Leakage

Split before fitting:

- scalers
- encoders
- imputers
- selectors
- transformations

Fit preprocessing only on training data.

Keep the test set untouched until final
evaluation.

## Preprocessing

Use clear sklearn preprocessing when
appropriate.

ColumnTransformer and Pipeline are allowed.

Do not manually duplicate preprocessing in
Streamlit when a saved preprocessor exists.

## Classification Metrics

Where appropriate use:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve

Do not rely only on accuracy.

## Regression Metrics

Where appropriate use:

- MAE
- RMSE
- R2

Add other metrics only when useful.

## Single Model Requests

If one model is requested:

- build only that model
- do not compare algorithms
- do not add baselines
- do not create comparison tables

Respect scope.

## Neural Networks

When architecture is specified, follow it
exactly.

Example:

Dense(64)
Dense(32)
Dense(16)
Dense(1)

Do not silently add hidden layers.

For binary classification normally use:

- ReLU hidden activations
- Sigmoid output
- Binary Crossentropy
- Adam

Use EarlyStopping when appropriate.

## Visualizations

Visualizations must be created inside the
notebook from notebook variables.

The notebook itself must generate:

- training curves
- confusion matrix
- ROC curve
- PR curve
- feature importance
- EDA charts
- metric tables

Do not create them outside the notebook
and merely load them back.

## Reports

Reports are allowed.

Examples:

reports/
`-- figures/

Useful outputs may include:

- final figures
- metrics tables
- feature importance tables
- comparison tables when requested
- presentation-ready outputs

But the notebook must create the original
results first.

Correct flow:

calculate in notebook
-> display in notebook
-> interpret in Markdown
-> optionally save a copy

External reports are outputs, not inputs.

Do not use pre-generated PNG, CSV, or JSON
files as a substitute for notebook logic.

## Feature Importance

Use a technically valid method.

Examples:

- native tree importance
- coefficients
- permutation importance
- SHAP only if actually implemented

For neural networks, do not use fake
tree-based feature_importances_.

Explain that predictive importance does
not prove causality.

## Model Saving

Save models inside the project.

Preferred:

models/

Examples:

models/customer_churn_nn.keras

models/preprocessor.joblib

models/feature_names.joblib

Do not save project models into a shared
repository-wide models folder.

## Artifact Validation

Reload important artifacts after saving.

Verify:

- model loads
- preprocessor loads
- feature order is correct
- predictions remain consistent

Do not create a separate validation report
just for bookkeeping.

## Streamlit

The Streamlit app is for inference.

The app should:

1. load saved model
2. load preprocessing
3. collect user input
4. transform input
5. predict
6. show result
7. show useful interpretation

The app must not train the model.

The notebook creates the artifacts used by
the app.

## File Creation Discipline

Do not create files just because a typical
ML template contains them.

Before creating a Python file, ask:

Does this file provide genuinely reusable
functionality?

If no, do not create it.

A complete project does not automatically
need:

- train.py
- config.py
- runner.py
- experiment manager
- reporting script
- CLI entry point

## Final Summary

Finish the notebook with Markdown.

Summarize:

- data preparation
- preprocessing
- model
- training
- evaluation
- feature importance
- reports
- saved model
- Streamlit integration
- limitations

## Final Check

Before completion verify:

- notebook contains the real training logic
- no unnecessary train.py exists
- code layout is clean
- normal cells are small
- variables use preferred naming
- comments are short and useful
- Markdown is professional
- no data leakage exists
- figures are generated in notebook
- reports are optional copies
- saved model is project-local
- app does not retrain model
- unrelated files were not modified

<!-- STREAMLIT_STYLE_START -->

# Streamlit Application Style

These rules define the preferred Streamlit
design and code organization for ML projects.

## Core Streamlit Rule

The Streamlit application is the deployment
and inference layer.

The notebook is responsible for:

- training
- preprocessing fitting
- evaluation
- metrics
- feature importance
- model saving

The Streamlit app is responsible for:

- loading saved artifacts
- collecting user input
- inference
- prediction explanation
- metrics presentation
- feature importance presentation
- project information

Never train or refit the model inside
Streamlit.

## App Structure

For project-specific applications prefer:

project_name/
|
|-- app/
|   |-- app.py
|   `-- utils/
|       `-- ui_components.py
|
|-- data/
|-- models/
|-- notebooks/
|-- reports/
|-- README.md
`-- requirements.txt

Only create `utils/` when reusable UI code
actually exists.

Do not create unnecessary modules.

## Main app.py Philosophy

Keep `app.py` relatively clean.

The main file should focus on:

- page configuration
- artifact loading
- page layout
- input form
- prediction
- displaying results

Reusable UI styling may live in:

app/utils/ui_components.py

Avoid putting hundreds of lines of CSS,
HTML and repeated layout code directly
inside app.py.

## Page Configuration

Configure Streamlit near the beginning.

Preferred style:

st.set_page_config(
    page_title="Project Name",
    page_icon="...",
    layout="wide",
)

Use:

layout="wide"

for ML dashboards unless the user requests
another layout.

Choose a page title specific to the project.

## UI Components

When the app has a developed visual design,
prefer reusable components such as:

apply_custom_css()

render_sidebar()

render_header()

render_about_section()

Additional reusable components are allowed
when useful, for example:

render_metric_card()

render_prediction_card()

render_model_info()

Do not create components for tiny one-line
Streamlit calls.

Use components only when they improve
readability or reuse.

## UI Components Import Style

A project may use a clean import such as:

from utils.ui_components import (
    apply_custom_css,
    render_sidebar,
    render_header,
    render_about_section,
)

If the app folder requires path handling,
keep it minimal.

Do not create complicated import systems.

Do not modify sys.path unless the actual
project structure requires it.

Prefer normal package imports whenever
possible.

## Custom CSS

Custom CSS is encouraged when it improves
the interface.

Use it for:

- cards
- spacing
- typography
- metric containers
- prediction panels
- sidebar improvements
- section containers

Keep custom CSS centralized when possible.

Preferred:

apply_custom_css()

Avoid scattering large CSS blocks throughout
multiple sections of app.py.

## Header

Use a clear custom header near the top.

The header should communicate:

- project name
- short description
- page purpose

Example concept:

render_header(
    "Customer Churn Prediction",
    "Predict customer churn using the "
    "trained neural network.",
    "Prediction Dashboard"
)

Keep descriptions concise.

## Sidebar

Use the sidebar for secondary information
and navigation where useful.

Suitable sidebar content includes:

- project navigation
- model information
- prediction threshold
- dataset information
- app sections
- about information

Do not move the primary prediction form into
the sidebar unless that layout genuinely
works better.

## Cards

Prefer clean cards for important sections.

Cards may be created with:

st.markdown(
    "...",
    unsafe_allow_html=True
)

Useful card types:

- project overview
- prediction result
- probability result
- model information
- risk interpretation
- important features
- instructions

Do not turn every paragraph into a card.

Use cards for visual hierarchy.

## HTML in Streamlit

HTML is allowed for layout and presentation.

Use it intentionally.

Good uses:

- styled cards
- custom headings
- compact descriptions
- prediction status panels

Avoid giant HTML applications embedded
inside Streamlit.

Streamlit widgets should remain responsible
for application interaction.

## Main Page Layout

A project ML application should normally
follow this flow:

1. Page configuration
2. Custom CSS
3. Sidebar
4. Header
5. Project overview
6. Prediction form
7. Prediction result
8. Probability / risk level
9. Explanation
10. Model metrics
11. Feature importance
12. Model information
13. About project

Adapt sections to the project.

Do not force irrelevant sections.

## Project Overview

Near the beginning, include a short overview.

Explain:

- what is predicted
- what model is being used
- what the user should enter

Keep it compact.

Do not reproduce the README.

## Input Form

Organize prediction inputs cleanly.

Prefer:

with st.form(...):

when several related inputs are submitted
together.

Use columns to organize related features.

Example concept:

col1, col2 = st.columns(2)

Group related fields.

Do not create one long vertical stack when
columns provide better readability.

## Widgets

Choose widgets according to the feature.

Examples:

st.number_input()

st.selectbox()

st.slider()

st.radio()

st.toggle()

Use sensible defaults based on the dataset
when known.

Do not invent impossible ranges.

Use original feature names internally.

UI labels may be made more readable.

## Prediction Workflow

Prediction must follow:

user input
-> one-row DataFrame
-> saved preprocessor
-> saved model
-> probability
-> predicted class

Do not manually duplicate encoding.

Do not manually duplicate scaling.

Do not rebuild preprocessing inside the app.

## Model Loading

Use cached resource loading.

Preferred pattern:

@st.cache_resource
def load_model():
    ...

Load:

- model
- preprocessor
- feature schema

only once when possible.

Do not repeatedly reload artifacts on every
widget interaction.

## Data / Metrics Loading

Use caching where useful.

For lightweight static data:

@st.cache_data

may be used.

Do not cache everything automatically.

## Prediction Result

Prediction results should be visually clear.

Display:

- predicted class
- probability
- simple interpretation

For classification, distinguish outcomes
visually.

Example concept:

Low Risk

or:

High Churn Risk

Do not display only:

Prediction: 1

Translate the prediction into human-readable
meaning.

## Probability

Show probability separately from the class.

Examples:

Churn Probability: 73.4%

Retention Probability: 26.6%

Use a progress indicator, metric card or
styled result when appropriate.

Do not present probability as certainty.

## Risk Interpretation

If useful, convert probability into simple
risk bands.

Example:

Low
Moderate
High

Only use thresholds that are clearly defined.

Do not create scientifically unsupported
risk labels.

## Metrics Section

Display final saved model metrics.

For classification this may include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

Prefer:

st.metric()

inside columns for headline metrics.

Example concept:

col1, col2, col3 = st.columns(3)

Do not recalculate test metrics inside
Streamlit.

Load notebook-generated results.

## Evaluation Visualizations

The app may display evaluation visualizations
generated from notebook results.

Examples:

- confusion matrix
- ROC curve
- Precision-Recall curve
- training history

The notebook remains responsible for the
original evaluation.

Streamlit is displaying the final results.

## Feature Importance

Provide a clear feature-importance section
when available.

Feature importance should come from the
method actually implemented in the notebook.

Examples:

- native tree importance
- permutation importance
- coefficients
- SHAP

Do not fabricate feature importance.

Prefer an interactive or clean bar chart.

Also show a small top-features table when
useful.

## Prediction Explanation

Where technically valid, explain why the
prediction may have been influenced.

Global feature importance may be shown.

Customer-specific explanations should only
be shown when a valid local explanation
method exists.

Never fabricate:

- SHAP values
- contribution percentages
- causal effects

## Model Information

Include a concise model information section.

Useful information:

- model type
- target
- threshold
- important preprocessing
- neural network architecture
- evaluation metric

Do not dump the entire training configuration.

## Neural Network Information

For NN apps, architecture can be shown
cleanly.

Example:

Input

-> Dense 64

-> Dense 32

-> Dense 16

-> Sigmoid Output

Keep the explanation readable for a general
viewer.

## About Section

End with a project-specific About section.

A reusable function such as:

render_about_section()

is preferred when the same design is used
across multiple applications.

The About section may contain:

- project purpose
- technologies
- model type
- author information

Keep it concise.

## Visual Hierarchy

The interface should feel like a dashboard,
not a raw Streamlit script.

Use:

- wide layout
- sections
- cards
- columns
- spacing
- clear headings
- metric blocks

Avoid:

- giant uninterrupted text
- dozens of separators
- excessive emojis
- random widget placement
- inconsistent spacing
- default Streamlit look everywhere

## Emoji Rule

Emojis are allowed in the Streamlit UI.

They may be useful for:

- page icon
- prediction status
- navigation
- cards

Use them selectively.

Do not fill every heading with emojis.

This UI rule does not mean notebook code or
Markdown should become emoji-heavy.

## Color and Styling

Use a consistent design.

Prefer one coherent visual system.

Do not use random colors for every card.

Prediction states may use different styling
when useful.

Maintain readable contrast.

## Responsiveness

Use Streamlit columns sensibly.

Avoid overly wide individual components.

Do not hard-code huge fixed widths unless
required.

The app should remain usable on normal
desktop displays.

## File Organization

Preferred when reusable UI exists:

app/
|
|-- app.py
`-- utils/
    |-- __init__.py
    `-- ui_components.py

Keep prediction/model utilities separate
only when genuinely necessary.

Do not create:

- ui_manager.py
- layout_engine.py
- dashboard_factory.py
- component_registry.py

for a simple project.

Avoid architecture for architecture's sake.

## Project-Specific App

When building a standalone project app,
do not automatically integrate it into a
global portfolio dashboard.

Each ML project may have its own app.

Example:

project_name/
`-- app/
    `-- app.py

Only integrate into a shared dashboard if
the user explicitly requests it.

## Clean app.py

Prefer readable spacing.

Separate logical sections.

Use short comments such as:

# Load model artifacts

# Build prediction form

# Run model inference

# Display model metrics

# Show feature importance

Do not write unnecessary comments.

Do not write giant blocks with everything
touching everything else.

## Streamlit Code Length

The notebook 7-line rule does NOT apply
strictly to Streamlit files.

Streamlit code should prioritize:

- clean layout
- readable structure
- logical grouping

Still avoid excessively long functions.

Break reusable UI into components when it
clearly improves readability.

## Streamlit Final Validation

Before completion verify:

- page uses wide layout
- title is project-specific
- app loads saved artifacts
- app does not train
- preprocessing is not duplicated
- feature order is preserved
- prediction works
- probability is displayed
- result is human-readable
- metrics are visible
- feature importance is valid
- UI has clear hierarchy
- custom CSS is consistent
- app.py is not unnecessarily huge
- reusable UI is separated when useful
- no unnecessary Python files exist
- paths are project-relative
- app matches the real project
- unrelated projects are untouched

<!-- STREAMLIT_STYLE_END -->


## Code Comments and Spacing

Every normal notebook code cell must start
with one short useful comment.

Example:

# Plot churn distributions

or:

# Prepare training features

or:

# Evaluate model performance

Do not start normal code cells directly with
code.

Comments should explain the logical purpose
of the block.

Do not comment every single line.

Bad:

# Create figure
figure = ...

# Flatten axes
axes = ...

# Plot values
data[...]

Good:

# Visualize churn patterns
figure, axes = ...

Every logical code block must be separated
with a blank line.

Use blank lines between:

- setup and transformation
- loops and plotting
- data preparation and output
- calculations and visualization
- model prediction and evaluation
- table creation and display

Do not write visually compressed cells.

Bad:

figure, axes = plt.subplots(...)
axes = axes.ravel()
data[...]
for ...:
    ...
for ...:
    ...
figure.suptitle(...)
plt.show()

Preferred style:

# Visualize churn patterns
figure, axes = plt.subplots(
    3, 3,
    figsize=(15, 11)
)

axes = axes.ravel()

data["Exited"].value_counts() \
    .sort_index() \
    .plot.bar(ax=axes[0])

axes[0].set(
    title="Churn Distribution",
    xlabel="Exited",
    ylabel="Customers"
)

# Compare numerical features
for axis, column in zip(
    axes[1:5],
    ["Age", "Balance",
     "CreditScore", "Tenure"]
):
    data.boxplot(
        column=column,
        by="Exited",
        ax=axis
    )

    axis.set_title(
        f"{column} vs Churn"
    )

# Compare categorical features
for axis, column in zip(
    axes[5:],
    ["Geography", "Gender",
     "NumOfProducts",
     "IsActiveMember"]
):
    churn_rate = (
        data.groupby(column)["Exited"]
        .mean()
    )

    churn_rate.plot.bar(
        ax=axis
    )

    axis.set(
        title=f"{column} vs Churn",
        ylabel="Churn Rate"
    )

figure.suptitle(
    "Customer Churn Data Understanding",
    y=1.02
)

figure.tight_layout()
plt.show()

### Layout Rules

Prefer vertical formatting over long horizontal
lines.

Break function calls into multiple lines when
they contain several arguments.

Prefer:

model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    )
)

instead of:

model.fit(X_train, y_train,
validation_data=(X_val, y_val))

Use parentheses for multiline expressions.

Avoid excessive backslashes.

Keep related lines visually grouped.

### Cell Structure

A normal cell should look like:

1. short comment
2. first logical block
3. blank line
4. second logical block
5. blank line
6. output/display when needed

The goal is readable visual rhythm.

Do not pack unrelated operations together
just to reduce the number of cells.

### Visualization Cells

Visualization cells may exceed the normal
7-line rule.

But they must still:

- start with a useful comment
- use blank lines between logical sections
- use clean multiline formatting
- separate plotting groups clearly
- avoid long compressed loops
- use descriptive variable names
- end with a clean display call

### Spacing Priority

For notebook code, layout quality has higher
priority than minimizing vertical space.

It is acceptable for a visualization cell to
be longer when the code becomes easier to read.

Do not optimize for the fewest possible lines.

Optimize for readability.


## Mandatory Code Style Examples

Use these examples as the visual reference
for notebook code.

The goal is not only correct code.

The goal is code that visually matches this
style:

- useful comment at the start of the cell
- blank lines between logical blocks
- short readable sections
- clean multiline formatting
- no compressed code
- comments for logical sections, not every line

# Example 1 - Data Cleaning

Bad:

identifier_columns = ["RowNumber", "CustomerId", "Surname"]
target = "Exited"
modeling_data = data.drop(columns=identifier_columns)
print("Dropped:", identifier_columns)
print("Modeling columns:", modeling_data.columns.tolist())

Good:

# Remove identifier columns
identifier_columns = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

modeling_data = data.drop(
    columns=identifier_columns
)

# Check the remaining features
modeling_data.columns.tolist()

Notice:

- comment before the first logical block
- list written vertically
- blank line before the next operation
- second comment before a new purpose
- unnecessary print statements removed

# Example 2 - Missing Values

Bad:

missing = data.isna().sum()
missing = missing[missing > 0]
print(missing)

Good:

# Check missing values
missing_values = (
    data.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_values[
    missing_values > 0
]

# Example 3 - Duplicate Rows

Bad:

print(data.duplicated().sum())
data = data.drop_duplicates()
print(data.shape)

Good:

# Check duplicate rows
duplicate_count = (
    data.duplicated()
    .sum()
)

duplicate_count

Then use a separate cell:

# Remove duplicate rows
data = (
    data.drop_duplicates()
    .reset_index(drop=True)
)

data.shape

Do not combine checking and modifying data
in one compressed block when separating them
makes the workflow clearer.

# Example 4 - Train Test Split

Bad:

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

Good:

# Split the dataset
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

# Example 5 - Preprocessing

Bad:

X_train_processed = preprocessor.fit_transform(X_train).astype("float32")
X_validation_processed = preprocessor.transform(X_validation).astype("float32")
X_test_processed = preprocessor.transform(X_test).astype("float32")
feature_names = preprocessor.get_feature_names_out().tolist()
print(X_train_processed.shape, X_validation_processed.shape)
print(X_test_processed.shape, len(feature_names))

Good:

# Fit the preprocessing pipeline
X_train_processed = (
    preprocessor
    .fit_transform(X_train)
    .astype("float32")
)

Then a separate cell:

# Transform validation data
X_validation_processed = (
    preprocessor
    .transform(X_validation)
    .astype("float32")
)

Then:

# Transform test data
X_test_processed = (
    preprocessor
    .transform(X_test)
    .astype("float32")
)

Then:

# Get transformed feature names
feature_names = (
    preprocessor
    .get_feature_names_out()
    .tolist()
)

This is preferred over putting training,
validation, test transformation, feature names,
and multiple prints inside the same cell.

# Example 6 - Model Definition

Bad:

model = Sequential([
Dense(64, activation="relu"),
Dense(32, activation="relu"),
Dense(16, activation="relu"),
Dense(1, activation="sigmoid")
])

Good:

# Build the neural network
model = Sequential([
    Dense(
        64,
        activation="relu"
    ),
    Dense(
        32,
        activation="relu"
    ),
    Dense(
        16,
        activation="relu"
    ),
    Dense(
        1,
        activation="sigmoid"
    )
])

Visualization and model-definition cells may
exceed the normal line-length preference when
clean vertical formatting improves readability.

# Example 7 - Model Compile

Bad:

model.compile(optimizer="adam", loss="binary_crossentropy",
metrics=["accuracy", AUC(name="auc")])

Good:

# Configure model training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        AUC(name="auc")
    ]
)

# Example 8 - Model Training

Bad:

history = model.fit(X_train_processed, y_train,
validation_data=(X_validation_processed, y_validation),
epochs=100, batch_size=32, callbacks=[early_stopping],
verbose=1)

Good:

# Train the neural network
history = model.fit(
    X_train_processed,
    y_train,
    validation_data=(
        X_validation_processed,
        y_validation
    ),
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stopping
    ],
    verbose=1
)

# Example 9 - Predictions

Bad:

y_probability = model.predict(X_test_processed).ravel()
y_pred = (y_probability >= 0.5).astype(int)

Good:

# Generate test predictions
y_probability = (
    model
    .predict(
        X_test_processed,
        verbose=0
    )
    .ravel()
)

y_pred = (
    y_probability >= 0.5
).astype(int)

# Example 10 - Metrics

Bad:

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)
print(accuracy, precision, recall, f1, roc_auc)

Good:

# Calculate classification metrics
metrics = {
    "Accuracy": accuracy_score(
        y_test,
        y_pred
    ),
    "Precision": precision_score(
        y_test,
        y_pred
    ),
    "Recall": recall_score(
        y_test,
        y_pred
    ),
    "F1 Score": f1_score(
        y_test,
        y_pred
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        y_probability
    )
}

Then use another cell:

# Display model performance
metrics_df = pd.DataFrame(
    metrics.items(),
    columns=[
        "Metric",
        "Score"
    ]
)

metrics_df

# Example 11 - Visualization

Bad:

figure, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.ravel()
data["Exited"].value_counts().sort_index().plot.bar(ax=axes[0])
axes[0].set(title="Churn Distribution", xlabel="Exited", ylabel="Customers")
for axis, column in zip(axes[1:5], ["Age", "Balance", "CreditScore", "Tenure"]):
    data.boxplot(column=column, by="Exited", ax=axis)
    axis.set_title(f"{column} vs Churn")
for axis, column in zip(axes[5:], ["Geography", "Gender", "NumOfProducts", "IsActiveMember"]):
    data.groupby(column)["Exited"].mean().plot.bar(ax=axis)
    axis.set(title=f"{column} vs Churn", ylabel="Churn Rate")
figure.suptitle("Customer Churn Data Understanding", y=1.02)
figure.tight_layout()
plt.show()

Good:

# Visualize churn patterns
figure, axes = plt.subplots(
    3,
    3,
    figsize=(15, 11)
)

axes = axes.ravel()

# Plot target distribution
churn_counts = (
    data["Exited"]
    .value_counts()
    .sort_index()
)

churn_counts.plot.bar(
    ax=axes[0]
)

axes[0].set(
    title="Churn Distribution",
    xlabel="Exited",
    ylabel="Customers"
)

# Compare numerical features
numeric_features = [
    "Age",
    "Balance",
    "CreditScore",
    "Tenure"
]

for axis, column in zip(
    axes[1:5],
    numeric_features
):
    data.boxplot(
        column=column,
        by="Exited",
        ax=axis
    )

    axis.set_title(
        f"{column} vs Churn"
    )

# Compare categorical features
categorical_features = [
    "Geography",
    "Gender",
    "NumOfProducts",
    "IsActiveMember"
]

for axis, column in zip(
    axes[5:],
    categorical_features
):
    churn_rate = (
        data.groupby(column)["Exited"]
        .mean()
    )

    churn_rate.plot.bar(
        ax=axis
    )

    axis.set(
        title=f"{column} vs Churn",
        ylabel="Churn Rate"
    )

# Finalize the figure
figure.suptitle(
    "Customer Churn Data Understanding",
    y=1.02
)

figure.tight_layout()
plt.show()

The visualization cell can be longer.

The important point is visual separation into
clear logical sections.

# Example 12 - Saving a Figure

Bad:

plt.savefig(reports_dir / "roc_curve.png")
img = Image.open(reports_dir / "roc_curve.png")
display(img)

Good:

# Plot the ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title(
    "Neural Network ROC Curve"
)

plt.tight_layout()
plt.show()

If saving is useful, save from the same
notebook-generated figure.

Do not use an external image as the notebook's
main visualization.

# Example 13 - Feature Importance

Bad:

importance = permutation_importance(...)
pd.DataFrame(...).to_csv(...)
print("saved")

Good:

# Calculate permutation importance
importance_result = (
    permutation_importance(
        estimator,
        X_test_processed,
        y_test,
        scoring="roc_auc",
        random_state=RANDOM_STATE
    )
)

Then:

# Build feature importance table
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance":
        importance_result.importances_mean
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_importance.head(10)

Then visualize it in another cell.

# Example 14 - Saving the Model

Bad:

model.save("../artifacts/model.keras")
joblib.dump(preprocessor, "../artifacts/preprocessor.joblib")
joblib.dump(feature_names, "../artifacts/features.joblib")

Good:

# Create the model directory
models_path.mkdir(
    parents=True,
    exist_ok=True
)

Then:

# Save the trained model
model.save(
    models_path /
    "customer_churn_nn.keras"
)

Then:

# Save preprocessing artifacts
joblib.dump(
    preprocessor,
    models_path /
    "preprocessor.joblib"
)

joblib.dump(
    feature_names,
    models_path /
    "feature_names.joblib"
)

Separate important artifact operations when it
improves clarity.

# Example 15 - Preferred Visual Rhythm

The preferred code rhythm is:

# Short comment
first_block = (
    something
)

second_operation()

# New logical purpose
new_block = (
    something_else
)

display_result()

Not:

first_block = something
second_operation()
new_block = something_else
display_result()

Comments and blank lines should make the
logical structure visible before reading the
details of the code.

# Final Style Enforcement

When generating or repairing notebook code,
compare each cell against the examples above.

If a cell looks visually compressed compared
with these examples, reformat it.

Do not wait for the user to request formatting
after generation.

Formatting is part of the implementation.

<!-- MARKDOWN_STYLE_START -->

# Notebook Markdown Style

These rules define the preferred Markdown
style for ML notebooks.

## Core Markdown Rule

Markdown should organize and explain the
workflow.

It should NOT read like a textbook.

Do not describe code using phrases such as:

- This cell builds...
- This cell creates...
- This cell performs...
- The following cell will...
- In the code below...
- The following code block...

Never narrate the existence of a cell.

Explain the analytical step itself.

Bad:

This cell builds the numerical preprocessing
pipeline. Median imputation handles any missing
values while StandardScaler puts neural-network
inputs on a comparable scale.

Good:

## Numerical Preprocessing

Numerical features are imputed with the median
and scaled using `StandardScaler` before model
training.

## Markdown Length

For normal technical steps, Markdown should
usually contain:

- one clear heading
- one or two short sentences

Do not write a full paragraph when one sentence
explains the purpose.

Do not repeat information already obvious from
the heading.

Longer Markdown is allowed for:

- project overview
- business problem
- important modeling decisions
- interpretation
- final summary

## Heading Style

Use clear hierarchy.

Major notebook phase:

# Phase 3 — Data Cleaning

or:

## Phase 3: Data Cleaning

Keep one phase-heading convention consistent
inside the notebook.

Normal step:

## Missing Values

## Numerical Preprocessing

## Categorical Preprocessing

## Train, Validation, and Test Split

## Neural Network Architecture

## Model Evaluation

## Feature Importance

Small analytical subsection:

### Business Insight

### Recommendation

### Interpretation

Do not create huge titles for simple steps.

Bad:

## Comprehensive Numerical Feature
Preprocessing Pipeline Construction and
Transformation

Good:

## Numerical Preprocessing

## Preferred Writing Voice

Use direct professional English.

Prefer:

We check...

We remove...

We split...

We scale...

We train...

We evaluate...

This step checks...

The target variable is...

The model uses...

Avoid overly academic language.

Avoid unnecessary phrases such as:

- It is important to note that
- It should be mentioned that
- In order to
- In the following section
- As can be observed
- It can clearly be seen
- This cell is responsible for
- The purpose of this code cell is

Write the point directly.

## Before Code Cells

Every important code cell should have Markdown
before it.

The Markdown should explain:

1. what step is being performed
2. why it matters, only when the reason is not
   obvious

Do not explain every implementation detail.

The code and its comments handle implementation.

Markdown handles workflow and reasoning.

## Example 1 — Data Loading

Bad:

## Data Loading

This cell loads the customer churn dataset from
the configured data path into a pandas DataFrame
so that the following analysis can be performed.

Good:

## Data Loading

The customer churn dataset is loaded and checked
before starting the analysis.

## Example 2 — Missing Values

Bad:

## Missing Value Analysis

This cell calculates the number of missing values
for every feature in the dataset and displays the
result so that we can determine whether imputation
is required.

Good:

## Missing Values

We check the dataset for missing values before
applying any preprocessing.

## Example 3 — Identifier Columns

Bad:

## Identifier Column Removal

The following cell removes RowNumber, CustomerId,
and Surname because these columns are identifier
features that do not contain meaningful predictive
information for the classification model.

Good:

## Remove Identifier Columns

`RowNumber`, `CustomerId`, and `Surname` are
removed because they do not provide useful
predictive information.

## Example 4 — Target

Bad:

This cell defines Exited as the target variable
that will be predicted by the neural network.

Good:

## Target Variable

The target is `Exited`, where:

- `0` means the customer stayed.
- `1` means the customer churned.

## Example 5 — Data Split

Bad:

## Dataset Splitting Strategy

This cell divides the dataset into training,
validation, and testing partitions. Stratification
is applied so that the proportion of churned and
non-churned customers remains consistent across
all subsets.

Good:

## Train, Validation, and Test Split

The data is split using stratification to preserve
the churn distribution across all subsets.

## Example 6 — Numerical Pipeline

Bad:

This cell builds the numerical preprocessing
pipeline. Median imputation handles any missing
values while StandardScaler puts neural-network
inputs on a comparable scale.

Good:

## Numerical Preprocessing

Numerical features are imputed with the median
and scaled using `StandardScaler`.

## Example 7 — Categorical Pipeline

Bad:

This cell builds the categorical preprocessing
pipeline. It imputes the most frequent category
and one-hot encodes it while safely ignoring
unseen app categories.

Good:

## Categorical Preprocessing

Categorical features are filled with the most
frequent value and encoded using
`OneHotEncoder`.

Unknown categories are safely ignored during
inference.

Only use the second sentence when this detail is
actually relevant.

## Example 8 — Preprocessor

Bad:

This cell combines both the numerical and
categorical pipelines into a ColumnTransformer,
which ensures that each feature type receives the
correct preprocessing operation.

Good:

## Combine Preprocessing Steps

Numerical and categorical transformations are
combined using a `ColumnTransformer`.

## Example 9 — Model Architecture

Bad:

This cell creates a Sequential neural network
consisting of three hidden layers with 64, 32,
and 16 neurons respectively using ReLU activation,
followed by a sigmoid output layer for binary
classification.

Good:

## Neural Network Architecture

The model contains three hidden layers:

- 64 neurons
- 32 neurons
- 16 neurons

ReLU is used in the hidden layers and Sigmoid is
used for the binary output.

## Example 10 — Model Compilation

Bad:

This cell compiles the neural network using Adam
as the optimizer and Binary Crossentropy as the
loss function because this is a binary
classification problem.

Good:

## Model Configuration

The network uses Adam optimization and Binary
Crossentropy loss for binary classification.

## Example 11 — Training

Bad:

The following cell trains the neural network on
the training dataset while monitoring validation
performance and using EarlyStopping to prevent
overfitting.

Good:

## Model Training

The model is trained using the validation set to
monitor generalization.

`EarlyStopping` is used to reduce overfitting.

## Example 12 — Training Curves

Bad:

This cell visualizes training and validation loss
and accuracy across epochs to allow us to inspect
the model's learning behavior and identify signs
of overfitting.

Good:

## Training History

Training and validation curves are used to check
model convergence and possible overfitting.

## Example 13 — Evaluation

Bad:

This section evaluates the trained neural network
using multiple classification metrics because
accuracy alone may not provide a complete picture
of model performance.

Good:

## Model Evaluation

The final model is evaluated on the untouched
test set using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

## Example 14 — Confusion Matrix

Bad:

The confusion matrix provides a detailed view of
correct and incorrect classifications across both
target classes and helps identify false positives
and false negatives.

Good:

## Confusion Matrix

The confusion matrix shows how the model performs
across both churn classes.

Do not explain basic metric theory unless the
project specifically requires it.

## Example 15 — ROC Curve

Bad:

The ROC curve illustrates the relationship
between the true positive rate and false positive
rate across different classification thresholds.

Good:

## ROC Curve

The ROC curve is used to evaluate how well the
model separates churned and retained customers.

## Example 16 — Threshold Analysis

Bad:

# Classification Threshold Analysis

The probability threshold controls the trade-off
between precision and recall. Lower thresholds
generally identify more churners and increase
recall, but they may also produce more false
positives. The following table compares several
thresholds so that model behavior can be aligned
with retention capacity and intervention cost.

Good:

## Classification Threshold Analysis

Different probability thresholds are tested to
compare Precision, Recall, and F1 Score.

This helps select a better balance for churn
detection.

Do not turn a simple threshold step into a long
theoretical discussion.

## Example 17 — Feature Importance

Bad:

This cell calculates permutation importance by
randomly shuffling each transformed input feature
and measuring the resulting change in ROC-AUC.

Good:

## Feature Importance

Permutation Importance is used because the neural
network does not provide native feature
importance.

Features are ranked by their effect on ROC-AUC.

## Example 18 — Model Saving

Bad:

This cell serializes the final model and
preprocessing artifacts to disk so that they can
later be loaded by the Streamlit application
without retraining.

Good:

## Save Model Artifacts

The trained model and fitted preprocessing
objects are saved for Streamlit inference.

## Example 19 — Artifact Validation

Bad:

This cell reloads all serialized files and
verifies that the resulting predictions match
those generated before model serialization.

Good:

## Artifact Validation

The saved model and preprocessing artifacts are
reloaded to confirm that inference still works
correctly.

## Explanation Depth

Do not explain common Python or sklearn behavior
unless that behavior affects a modeling decision.

For example:

Do not explain what `.fit()` means.

Do explain why preprocessing is fitted only on
training data.

Do not explain what `plt.show()` does.

Do explain what an important visualization tells
us.

Do not explain basic DataFrame syntax.

Do explain why a column is removed.

## Theory Rule

Keep theory short.

Normal project notebooks are implementation and
analysis notebooks, not lectures.

Do not write textbook definitions before:

- StandardScaler
- OneHotEncoder
- train_test_split
- Dense layers
- Adam
- confusion matrix
- ROC curve

unless the user specifically asks for educational
theory.

Explain only the decision that matters to the
project.

## Bullets

Use bullet lists when several items belong
together.

Good:

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Avoid putting the same information into one long
sentence.

## Business Analysis Style

For business-oriented EDA, use the established
structure when useful:

## Business Question

### Business Question

What relationship are we investigating?

### Hypothesis

State the expected relationship briefly.

### Variables

- `Feature`
- `Target`

### Visualization

State the selected visualization briefly.

After the plot:

### Business Insight

Explain the observed pattern.

### Recommendation

Give a practical recommendation when the project
requires one.

Do not use this structure for every technical ML
step.

## Phase Introduction

At the beginning of a major phase, use a short
introduction.

Example:

# Phase 4 — Data Cleaning

After assessing the dataset, the identified data
quality issues are handled one step at a time.

The cleaned data is validated before moving to
feature preparation.

Do not write four or five paragraphs before a
normal phase.

## Phase Summary

For important phases, a short summary is allowed.

Preferred style:

# Phase 4 Summary

The main cleaning steps are complete:

- Missing values handled
- Duplicate rows removed
- Identifier columns removed
- Data types verified

The cleaned dataset is now ready for feature
preparation.

Do not create a summary after every small step.

## Visualization Interpretation

After an important visualization, use short
interpretation Markdown.

Good:

### Interpretation

Customers with three or more products show a
higher observed churn rate than customers with
one or two products.

Avoid:

The visualization above clearly demonstrates an
interesting and potentially significant pattern
which may suggest that...

State the finding directly.

## Metric Interpretation

Do not describe every metric definition.

Interpret the result.

Good:

### Model Performance

The model achieves strong ROC-AUC performance,
but Recall remains lower than Precision.

This means the model identifies churn risk
reasonably well but still misses part of the
positive class.

## Final Summary

The final notebook summary should be structured
and concise.

Preferred sections:

# Final Summary

## Data Preparation

Briefly summarize cleaning and preprocessing.

## Model

State the final architecture or algorithm.

## Evaluation

Summarize the final metrics and the main
performance observation.

## Feature Importance

State the most important model drivers and the
interpretability method.

## Deployment

State which artifacts were saved and how the
Streamlit app uses them.

Avoid repeating the notebook step by step.

## Forbidden Markdown Style

Do NOT generate normal Markdown that sounds like
an AI tutorial.

Avoid:

"This cell..."

"The following cell..."

"In this cell..."

"The code below..."

"In the next code block..."

"We will now proceed to..."

"The purpose of this cell is..."

"This implementation demonstrates..."

"This approach ensures a robust and scalable
machine learning workflow..."

Avoid inflated language such as:

- robust
- comprehensive
- sophisticated
- production-grade
- state-of-the-art
- highly scalable

unless the statement is technically necessary
and supported.

## Final Markdown Check

Before finishing a notebook, review the Markdown.

Check that:

- no normal section starts with "This cell"
- headings are short and descriptive
- technical explanations are concise
- Markdown explains decisions, not syntax
- bullets replace unnecessary long sentences
- theory is limited
- phase introductions are short
- interpretation states actual findings
- Markdown does not sound like a tutorial
- repeated explanations are removed
- the notebook reads like a clean project
  created by a student/ML engineer

<!-- MARKDOWN_STYLE_END -->
---

# Project README Style

## Core Rule

Write the README from the real project.

Before writing, inspect only the files needed
to understand the project.

Normally inspect:

- project structure
- main notebook
- app file
- requirements.txt
- saved models
- reports
- important artifacts

Do not invent files, metrics, models, paths,
commands, results, or technologies.

If something is not verified, omit it.

## Writing Style

Use professional and simple English.

The README should be:

- clear
- concise
- technical
- portfolio-ready
- easy to scan
- factual

Avoid:

- long academic paragraphs
- marketing language
- fake production claims
- unnecessary emojis
- generic AI wording
- repeated information

## Title

Start with a direct project title.

Examples:

Bank Customer Churn Prediction

Customer Churn Neural Network

Employee Attrition Analysis

Do not create complicated titles.

## Project Overview

Use this section:

Project Overview

Explain briefly:

- what the project does
- what problem it solves
- what dataset/domain it uses
- what model or analysis is used
- what the project delivers

Keep this section short.

## Business Problem

For business-oriented projects, include a
Business Problem section.

Explain the real problem in a short paragraph.

Do not force this section into a project where
it is not useful.

## Business Objectives

When relevant, use a Business Objectives section.

Prefer bullets.

Example:

- Understand customer churn patterns
- Identify important churn factors
- Build a predictive classification model
- Provide interpretable results

Only include objectives supported by the
project.

## Dataset

Use a short dataset section.

Include only verified information such as:

- dataset name
- target variable
- target meaning
- main domain
- data source if known

Example target meaning:

Exited = 1: customer churned

Exited = 0: customer stayed

Do not invent dataset dimensions or source.

## Project Workflow

Show the actual workflow.

For ML projects prefer a numbered list.

Example:

1. Data loading
2. Data understanding
3. Data cleaning
4. Exploratory data analysis
5. Feature engineering
6. Preprocessing
7. Model training
8. Model evaluation
9. Feature importance
10. Model saving
11. Streamlit integration

Do not add phases that do not exist.

For business EDA projects, a vertical flow is
also acceptable:

Business Understanding
        ↓
Data Understanding
        ↓
Data Quality Assessment
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Business Insights
        ↓
Recommendations

## Project Structure

For projects with multiple files, include the
real structure.

Preferred example:

    project_name/
    │
    ├── app/
    ├── data/
    ├── models/
    ├── notebooks/
    ├── reports/
    ├── src/
    ├── .gitignore
    ├── README.md
    └── requirements.txt

The structure must match the real project.

Do not include:

- venv
- .agents
- cache folders
- temporary files
- unrelated project folders

Do not create fake files just to make the
structure look professional.

## Data Cleaning

When relevant, summarize only what was
actually done.

Example:

- Handled missing values
- Removed duplicates
- Reviewed invalid values
- Removed identifier columns
- Reviewed outliers
- Removed constant features

Do not claim cleaning steps that were not used.

## Exploratory Data Analysis

For EDA projects summarize meaningful questions
or analysis areas.

Example:

- customer churn distribution
- age and churn
- balance and churn
- geography and churn
- activity status and churn

Do not describe every single plot.

## Feature Engineering

Include only when used.

List important engineered features.

Do not claim that a feature improved performance
unless that was actually tested.

## Preprocessing

For ML projects summarize the real preprocessing.

Examples:

- missing-value handling
- One-Hot Encoding
- StandardScaler
- train/test split
- stratification
- ColumnTransformer

Mention leakage-safe preprocessing when true.

## Model

Use a clear model section.

Example:

The project uses a Decision Tree Classifier.

For neural networks, use a compact architecture
block like this:

    Input
      → Dense(64, activation="relu")
      → Dense(32, activation="relu")
      → Dense(16, activation="relu")
      → Dense(1, activation="sigmoid")

Describe the actual model only.

Do not list models that were not trained.

## Model Evaluation

Show the actual metrics.

For classification:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For regression:

- MAE
- RMSE
- R²

If verified final values exist, use a small
table.

Example:

    Metric        Score
    Accuracy      0.87
    ROC-AUC       0.86

Never invent numbers.

If training is not finalized, describe the
evaluation method without fake scores.

## Feature Importance

Explain the actual interpretability method.

Examples:

- tree-based feature importance
- model coefficients
- permutation importance
- SHAP only if implemented

For neural networks, do not claim native tree
feature importance.

When appropriate mention that importance
indicates model sensitivity or association and
does not prove causality.

## Reports

A reports folder is allowed.

The README may mention useful outputs such as:

- saved figures
- metrics tables
- feature importance tables
- comparison tables
- final project report

Only mention files that really exist.

Do not make the README a full inventory of every
generated file.

## Streamlit Application

If a Streamlit app exists, explain briefly:

- what inputs it accepts
- what prediction it returns
- whether it shows probability
- whether it shows metrics
- whether it shows feature importance

Do not say the app retrains the model if it only
loads saved artifacts.

## Saved Models and Artifacts

When useful, show a small artifact summary.

Example:

    models/model.keras
    Final trained model

    models/preprocessor.joblib
    Saved preprocessing

Only include real files.

Do not list internal bookkeeping files unless
they matter.

## Tools & Technologies

List only technologies actually used.

Example:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

Do not add tools because they are popular.

## Installation

If requirements.txt exists, include:

    pip install -r requirements.txt

Do not add complicated setup unless required.

## Run the Notebook

If the notebook is the main workflow, say so.

Example:

    notebooks/main_notebook.ipynb

The README must reflect that the notebook
contains the actual training workflow.

Do not tell the user to run train.py unless such
a file intentionally exists.

## Run Streamlit

Use the actual app path.

Examples:

    streamlit run app/app.py

or:

    streamlit run app/streamlit_app.py

Verify the real path first.

## Repository-Level README

For a root portfolio README, use a different
structure.

Preferred order:

1. Repository title
2. Optional badges
3. Repository overview
4. Repository map
5. Projects table
6. Installation
7. Running applications
8. Working with notebooks
9. Main technologies
10. Author

Do not expose internal tools such as:

- .agents
- venv
- local Codex files
- internal repository graph statistics

unless explicitly requested.

## Badges

Badges are optional.

Useful badges may include:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Streamlit
- Jupyter

Do not add too many badges.

Do not add unused technologies.

## Author

End portfolio project READMEs with:

Ahmed Sameh Mohamed Zaky

Undergraduate Student, Pure Mathematics and
Computer Science

Use updated author wording if explicitly
provided later.

## Preferred README Order

### EDA Project

1. Title
2. Project Overview
3. Business Problem
4. Business Objectives
5. Dataset
6. Project Workflow
7. Project Structure
8. Data Cleaning
9. Exploratory Data Analysis
10. Feature Engineering
11. Tools & Technologies
12. Key Deliverables
13. Results
14. Author

### ML Project

1. Title
2. Optional badges
3. Project Overview
4. Dataset
5. Project Structure
6. Project Workflow
7. Preprocessing
8. Model
9. Evaluation
10. Feature Importance
11. Reports
12. Installation
13. Run Notebook
14. Run Streamlit
15. Author

### Neural Network + Streamlit

1. Title
2. Project Overview
3. Dataset and target
4. Project Structure
5. Workflow
6. Neural Network Architecture
7. Preprocessing
8. Evaluation
9. Feature Importance
10. Reports and Models
11. Installation
12. Run Notebook
13. Run Streamlit
14. Author

## README Length

Keep the README detailed enough to understand
the project but not so long that it reproduces
the notebook.

The README is a project map.

The notebook is the detailed analytical record.

## Final Check

Before finishing verify:

- title is correct
- overview matches the project
- project structure is real
- notebook path is correct
- app path is correct
- model name is correct
- target is correct
- metrics are verified
- technologies are actually used
- commands work with the real structure
- no fake files are mentioned
- no fake results are included
- no unrelated repository content appears
- README does not duplicate the full notebook
- author section is correct
