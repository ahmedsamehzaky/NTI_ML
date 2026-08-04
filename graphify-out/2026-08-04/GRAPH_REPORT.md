# Graph Report - C:\Users\soham\Desktop\ahmed\NTI-ML  (2026-08-04)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 287 nodes · 106 edges · 226 communities (15 shown, 211 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0990284f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- 1_Loan_Approval.py
- 04_RandomForest_HRAttrition/app/streamlit_app.py
- LoanAmount_Regression_Project/src/preprocessing.py
- Path
- build_preprocessor
- load_artifact
- load_artifacts
- load_artifacts
- load_artifacts
- AbstractValidator
- CancellationToken
- DbContext
- Dictionary
- ExitEventArgs
- IContainer
- IDbContextTransaction
- IDisposable
- IEntityTypeConfiguration
- IValueConverter
- ModelBuilder
- List
- IServiceCollection
- DateTime
- List
- DateTime
- List
- DateTime
- List
- DateTime
- DateTime
- List
- DateTime
- List
- PaymentMethod
- Task
- DateTime
- IReadOnlyList
- Task
- IReadOnlyList
- Task
- Task
- DateTime
- IReadOnlyList
- Task
- IReadOnlyList
- Task
- IReadOnlyList
- Task
- DateTime
- Task
- DateTime
- IReadOnlyList
- Task
- int
- string
- Task
- IMapper
- IReadOnlyList
- Task
- IMapper
- Task
- DateTime
- IMapper
- IReadOnlyList
- List
- Task
- IMapper
- IReadOnlyList
- Task
- IMapper
- IReadOnlyList
- Task
- DateTime
- IMapper
- Task
- DateTime
- IMapper
- IReadOnlyList
- Task
- DateTime
- DateTime
- DateTime
- DateTime
- ICollection
- DateTime
- ICollection
- DateTime
- ICollection
- DateTime
- DateTime
- ICollection
- DateTime
- ICollection
- PaymentMethod
- DateTime
- Task
- Expression
- Func
- IEnumerable
- IReadOnlyList
- Task
- Task
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- EntityTypeBuilder
- Task
- DbSet
- Task
- IServiceCollection
- IReadOnlyList
- Task
- IReadOnlyList
- Task
- IReadOnlyList
- Task
- DbSet
- Expression
- Func
- IEnumerable
- IReadOnlyList
- Task
- DateTime
- IReadOnlyList
- Task
- IReadOnlyList
- string
- Task
- Task
- Task
- IServiceCollection
- CultureInfo
- Type
- CultureInfo
- Type
- CultureInfo
- Type
- string
- bool
- Func
- string
- Task
- bool
- int
- RelayCommand
- string
- Task
- ObservableCollection
- RelayCommand
- string
- Task
- decimal
- int
- RelayCommand
- Task
- ObservableCollection
- RelayCommand
- string
- Task
- bool
- int
- RelayCommand
- string
- Task
- bool
- Func
- RelayCommand
- string
- Task
- decimal
- ObservableCollection
- RelayCommand
- string
- Task
- bool
- decimal
- int
- RelayCommand
- string
- Task
- ObservableCollection
- RelayCommand
- string
- Task
- bool
- int
- ObservableCollection
- RelayCommand
- string
- Task
- decimal
- int
- ObservableCollection
- PaymentMethod
- RelayCommand
- Task
- DateTime
- int
- RelayCommand
- string
- Task
- Fact
- Fact
- Fact
- Task
- object
- ObservableObject
- Profile
- ServiceProvider
- StartupEventArgs
- UserControl
- Window

## God Nodes (most connected - your core abstractions)
1. `load_artifact()` - 8 edges
2. `apply_custom_css()` - 7 edges
3. `render_sidebar()` - 7 edges
4. `render_header()` - 7 edges
5. `render_about_section()` - 7 edges
6. `get_model_path()` - 6 edges
7. `plot_feature_importance()` - 5 edges
8. `load_csv()` - 4 edges
9. `load_artifacts()` - 3 edges
10. `load_pipeline()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `load_csv()` --references--> `Path`  [EXTRACTED]
  04_RandomForest_HRAttrition/app/streamlit_app.py →   _Bridges community 1 → community 3_

## Import Cycles
- None detected.

## Communities (226 total, 211 thin omitted)

### Community 0 - "1_Loan_Approval.py"
Cohesion: 0.31
Nodes (11): get_model_path(), plot_feature_importance(), A helper mapping model names to their relative paths., apply_custom_css(), Renders the standard navigation sidebar and footer., Renders a modern hero section., Renders the About section at the bottom of pages., Injects custom CSS for a modern, professional AI dashboard aesthetic. (+3 more)

### Community 1 - "04_RandomForest_HRAttrition/app/streamlit_app.py"
Cohesion: 0.21
Nodes (8): employee_dataframe(), load_csv(), load_metadata(), load_pipeline(), probability_chart(), cache_resource, DataFrame, cache_data

### Community 2 - "LoanAmount_Regression_Project/src/preprocessing.py"
Cohesion: 0.24
Nodes (10): align_features(), build_feature_matrix(), cap_outliers_iqr(), clean_raw_dataframe(), DataFrame, Reusable preprocessing utilities for the loan amount regression project. These…, Apply structural cleaning to the raw loan approval dataset. Strips whitespace…, Apply Interquartile Range based winsorization to the specified numerical… (+2 more)

### Community 3 - "Path"
Cohesion: 0.20
Nodes (7): load_artifacts(), cache_resource, load_pipeline(), cache_resource, load_pipeline(), cache_resource, Path

### Community 4 - "build_preprocessor"
Cohesion: 0.50
Nodes (4): build_preprocessor(), clean_hr_data(), DataFrame, ColumnTransformer

### Community 5 - "load_artifact"
Cohesion: 0.40
Nodes (5): get_project_root(), load_artifact(), cache_resource, Loads a .pkl or .joblib file given a path relative to the repository root., Returns the path to the NTI-ML repository root.

## Knowledge Gaps
- **211 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_csv()` connect `04_RandomForest_HRAttrition/app/streamlit_app.py` to `Path`?**
  _High betweenness centrality (0.003) - this node is a cross-community bridge._
- **Why does `load_artifact()` connect `load_artifact` to `1_Loan_Approval.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._