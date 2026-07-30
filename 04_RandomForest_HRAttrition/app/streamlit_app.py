from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HR Employee Attrition Dashboard",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hr_attrition_classification_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "hr_attrition_model_metadata.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
EVALUATION_METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_metrics.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .app-header {
            padding: 1.25rem 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            margin-bottom: 1.25rem;
        }
        .app-header h1 {margin: 0; font-size: 2rem;}
        .app-header p {margin: 0.5rem 0 0 0; opacity: 0.82;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            padding: 1rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    return joblib.load(METADATA_PATH)


@st.cache_data
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


try:
    pipeline = load_pipeline()
except Exception as error:
    st.error("The trained pipeline could not be loaded.")
    st.exception(error)
    st.stop()

metadata = load_metadata()
feature_importance_df = load_csv(FEATURE_IMPORTANCE_PATH)
model_comparison_df = load_csv(MODEL_COMPARISON_PATH)
evaluation_metrics_df = load_csv(EVALUATION_METRICS_PATH)


def model_name() -> str:
    if metadata and metadata.get("selected_model"):
        return str(metadata["selected_model"])
    try:
        return pipeline.named_steps["model"].__class__.__name__
    except Exception:
        return "Classification Model"


def model_parameters() -> dict:
    try:
        params = pipeline.named_steps["model"].get_params()
        names = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "criterion",
            "bootstrap",
            "random_state",
        ]
        return {name: params.get(name) for name in names if name in params}
    except Exception:
        return {}


def employee_dataframe(values: dict) -> pd.DataFrame:
    frame = pd.DataFrame([values])
    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is not None:
        frame = frame.reindex(columns=list(expected))
    return frame


def probability_chart(attrition_probability: float, retention_probability: float):
    chart_df = pd.DataFrame(
        {
            "Outcome": ["Retention", "Attrition"],
            "Probability": [retention_probability * 100, attrition_probability * 100],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(chart_df["Outcome"], chart_df["Probability"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability Percentage")
    ax.set_title("Prediction Probability")
    for index, value in enumerate(chart_df["Probability"]):
        ax.text(min(value + 1, 94), index, f"{value:.2f}%", va="center")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def header(title: str, description: str):
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.sidebar.title("HR Attrition Project")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Employee Prediction",
        "Model Performance",
        "Feature Importance",
        "Model Comparison",
        "About Project",
    ],
)
st.sidebar.divider()
st.sidebar.success("Model loaded successfully")
st.sidebar.write(f"Model: **{model_name()}**")


if page == "Home":
    header(
        "HR Employee Attrition Dashboard",
        "A machine learning dashboard for estimating employee attrition risk and reviewing model results.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Primary Model", model_name())
    with col2:
        st.metric("Target", "Employee Attrition")
    with col3:
        st.metric("Positive Class", "Attrition = Yes")

    st.subheader("Application Capabilities")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            - Predict an employee's attrition probability.
            - Adjust the classification threshold.
            - View retention and attrition probabilities.
            - Download prediction results as CSV.
            """
        )
    with right:
        st.markdown(
            """
            - Review model performance reports.
            - Explore feature importance.
            - Compare classification models.
            - Inspect the trained model configuration.
            """
        )

    st.subheader("Model Configuration")
    params = model_parameters()
    if params:
        st.dataframe(
            pd.DataFrame({"Parameter": params.keys(), "Value": [str(v) for v in params.values()]}),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "This model is intended for analytical decision support only. It must not be used as the sole basis for employment decisions."
    )


elif page == "Employee Prediction":
    header(
        "Employee Attrition Prediction",
        "Enter an employee profile and generate a probability-based attrition estimate.",
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal Profile")
            age = st.number_input("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.selectbox(
                "Education Level",
                [1, 2, 3, 4, 5],
                index=2,
                format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}[x],
            )
            education_field = st.selectbox(
                "Education Field",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
            )
            distance_from_home = st.number_input("Distance From Home", 1, 29, 5)
            num_companies_worked = st.number_input("Number of Companies Worked", 0, 9, 2)
            total_working_years = st.number_input("Total Working Years", 0, 40, 10)
            training_times_last_year = st.number_input("Training Times Last Year", 0, 6, 3)
            work_life_balance = st.selectbox(
                "Work-Life Balance",
                [1, 2, 3, 4],
                index=2,
                format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x],
            )

        with col2:
            st.subheader("Job Profile")
            department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
            )
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
            job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
            performance_rating = st.selectbox("Performance Rating", [3, 4], index=0)

        with col3:
            st.subheader("Compensation and Tenure")
            monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=100)
            monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000, step=100)
            daily_rate = st.number_input("Daily Rate", 100, 1500, 800, step=10)
            hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
            percent_salary_hike = st.number_input("Percent Salary Hike", 11, 25, 14)
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
            years_at_company = st.number_input("Years at Company", 0, 40, 5)
            years_in_current_role = st.number_input("Years in Current Role", 0, 18, 3)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
            years_with_current_manager = st.number_input("Years With Current Manager", 0, 17, 3)

        st.divider()
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.01,
            help="Lower thresholds increase recall but may create more false positive alerts.",
        )
        submitted = st.form_submit_button("Predict Attrition Risk", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if years_at_company > total_working_years:
            errors.append("Years at Company cannot exceed Total Working Years.")
        if years_in_current_role > years_at_company:
            errors.append("Years in Current Role cannot exceed Years at Company.")
        if years_since_last_promotion > years_at_company:
            errors.append("Years Since Last Promotion cannot exceed Years at Company.")
        if years_with_current_manager > years_at_company:
            errors.append("Years With Current Manager cannot exceed Years at Company.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            values = {
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance_from_home,
                "Education": education,
                "EducationField": education_field,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies_worked,
                "OverTime": overtime,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times_last_year,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "YearsWithCurrManager": years_with_current_manager,
            }
            employee_df = employee_dataframe(values)

            try:
                probabilities = pipeline.predict_proba(employee_df)[0]
                retention_probability = float(probabilities[0])
                attrition_probability = float(probabilities[1])
                prediction = int(attrition_probability >= threshold)
            except Exception as error:
                st.error("The model could not generate a prediction.")
                st.exception(error)
            else:
                result_col, chart_col = st.columns(2)
                with result_col:
                    st.subheader("Prediction Result")
                    if prediction == 1:
                        st.error("Elevated attrition risk: the employee is predicted to leave.")
                    else:
                        st.success("Lower attrition risk: the employee is predicted to stay.")
                    st.metric("Attrition Probability", f"{attrition_probability * 100:.2f}%")
                    st.metric("Retention Probability", f"{retention_probability * 100:.2f}%")
                    st.caption(f"Threshold used: {threshold:.2f}")

                with chart_col:
                    probability_chart(attrition_probability, retention_probability)

                st.subheader("Submitted Employee Profile")
                display_df = employee_df.T.reset_index()
                display_df.columns = ["Feature", "Value"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                result_df = employee_df.copy()
                result_df["Prediction"] = "Likely to Leave" if prediction == 1 else "Likely to Stay"
                result_df["AttritionProbability"] = attrition_probability
                result_df["RetentionProbability"] = retention_probability
                result_df["ClassificationThreshold"] = threshold

                st.download_button(
                    "Download Prediction as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "employee_attrition_prediction.csv",
                    "text/csv",
                    use_container_width=True,
                )
                st.warning(
                    "This result is a statistical estimate and must not be used as the sole basis for an employment decision."
                )


elif page == "Model Performance":
    header(
        "Model Performance",
        "Review saved evaluation metrics and diagnostic figures generated by the notebook.",
    )

    if evaluation_metrics_df is not None and not evaluation_metrics_df.empty:
        row = evaluation_metrics_df.iloc[0]
        supported = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-Score"),
            ("roc_auc", "ROC-AUC"),
        ]
        available = [(c, label) for c, label in supported if c in row.index]
        columns = st.columns(len(available)) if available else []
        for container, (column, label) in zip(columns, available):
            with container:
                st.metric(label, f"{float(row[column]):.3f}")
        st.dataframe(evaluation_metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "reports/evaluation_metrics.csv was not found. Run the notebook export cells to generate it."
        )

    figures = [
        ("Confusion Matrix", FIGURES_DIR / "random_forest_confusion_matrix.png"),
        ("ROC Curve", FIGURES_DIR / "random_forest_roc_curve.png"),
        ("Precision-Recall Curve", FIGURES_DIR / "random_forest_precision_recall_curve.png"),
    ]
    found = False
    for title, path in figures:
        if path.exists():
            found = True
            st.subheader(title)
            st.image(str(path), use_container_width=True)
    if not found:
        st.info("No saved figures were found inside reports/figures.")


elif page == "Feature Importance":
    header(
        "Feature Importance",
        "Explore the variables used most heavily by the trained model.",
    )

    if feature_importance_df is None:
        st.warning("reports/feature_importance.csv was not found.")
    elif not {"feature", "importance"}.issubset(feature_importance_df.columns):
        st.error("The report must contain feature and importance columns.")
    else:
        maximum = min(30, len(feature_importance_df))
        count = st.slider("Number of Features", 5, maximum, min(20, maximum))
        selected = feature_importance_df.sort_values("importance", ascending=False).head(count)
        chart_col, table_col = st.columns([1.4, 1])
        with chart_col:
            plot_df = selected.sort_values("importance")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(plot_df["feature"], plot_df["importance"])
            ax.set_title("Random Forest Feature Importance")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with table_col:
            st.dataframe(selected, use_container_width=True, hide_index=True)
        st.caption("Feature importance describes model usage and does not establish causality.")


elif page == "Model Comparison":
    header(
        "Classification Model Comparison",
        "Compare Random Forest with the other classifiers evaluated in the notebook.",
    )

    if model_comparison_df is None:
        st.info("reports/model_comparison.csv was not found. Run the notebook comparison and export cells.")
    else:
        comparison = model_comparison_df.copy()
        comparison.columns = [c.strip().lower().replace(" ", "_") for c in comparison.columns]
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        metrics = [c for c in ["roc_auc", "f1_score", "accuracy", "precision", "recall"] if c in comparison.columns]
        if "model" in comparison.columns and metrics:
            selected_metric = st.selectbox(
                "Metric",
                metrics,
                format_func=lambda x: x.replace("_", " ").title(),
            )
            plot_df = comparison.sort_values(selected_metric)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(plot_df["model"], plot_df[selected_metric])
            ax.set_title(f"Model Comparison by {selected_metric.replace('_', ' ').title()}")
            ax.set_xlabel(selected_metric.replace("_", " ").title())
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            best = comparison.sort_values(selected_metric, ascending=False).iloc[0]
            st.success(
                f"Best model by {selected_metric.replace('_', ' ').title()}: "
                f"{best['model']} ({float(best[selected_metric]):.4f})"
            )


elif page == "About Project":
    header(
        "About the Project",
        "Technical architecture, expected files, and responsible-use guidance.",
    )

    st.subheader("Technical Architecture")
    st.markdown(
        """
        - Streamlit input interface and validation.
        - Numerical median imputation and scaling.
        - Categorical most-frequent imputation and one-hot encoding.
        - Saved end-to-end classification pipeline.
        - Probability prediction with an adjustable classification threshold.
        - Optional metrics, figures, feature importance, and model comparison reports.
        """
    )

    st.subheader("Expected Project Structure")
    st.code(
        """
04_RandomForest_HRAttrition/
├── app/
│   └── streamlit_app.py
├── data/
├── models/
│   ├── hr_attrition_classification_pipeline.joblib
│   └── hr_attrition_model_metadata.joblib
├── notebooks/
│   └── main_notebook.ipynb
├── reports/
│   ├── figures/
│   ├── evaluation_metrics.csv
│   ├── feature_importance.csv
│   └── model_comparison.csv
├── src/
├── README.md
└── requirements.txt
        """.strip(),
        language="text",
    )

    st.subheader("Run the Application")
    st.code("streamlit run app/streamlit_app.py", language="bash")

    st.subheader("Responsible Use")
    st.warning(
        "Attrition predictions may affect people and workplace decisions. Use the model only as carefully governed analytical support, review fairness across groups, and always require qualified human review."
    )
