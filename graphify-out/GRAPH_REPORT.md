# Graph Report - C:\Users\soham\Desktop\ahmed\NTI-ML  (2026-08-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 298 nodes · 77 edges · 237 communities (15 shown, 222 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `15aaf855`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- apply_custom_css
- random_forest_hr/app/streamlit_app.py
- regression/src/preprocessing.py
- Path
- nti_ml_app/utils/model_loader.py
- build_preprocessor
- load_artifacts
- load_artifacts
- load_artifacts
- cache_resource
- cache_resource
- cache_resource
- cache_resource
- DataFrame
- DataFrame
- cache_resource
- cache_resource
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
- cache_resource
- object
- ObservableObject
- Profile
- cache_resource
- DataFrame
- ServiceProvider
- StartupEventArgs
- UserControl
- Window

## God Nodes (most connected - your core abstractions)
1. `apply_custom_css()` - 5 edges
2. `render_sidebar()` - 5 edges
3. `render_header()` - 5 edges
4. `load_artifact()` - 4 edges
5. `load_csv()` - 4 edges
6. `get_project_root()` - 3 edges
7. `render_about_section()` - 3 edges
8. `load_artifacts()` - 3 edges
9. `build_preprocessor()` - 3 edges
10. `clean_raw_dataframe()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `load_csv()` --references--> `Path`  [EXTRACTED]
  projects/employee_attrition/random_forest_hr/app/streamlit_app.py →   _Bridges community 3 → community 1_

## Import Cycles
- None detected.

## Communities (237 total, 222 thin omitted)

### Community 0 - "apply_custom_css"
Cohesion: 0.22
Nodes (8): apply_custom_css(), Renders the standard navigation sidebar and footer., Renders a modern hero section., Renders the About section at the bottom of pages., Injects custom CSS for a modern, professional AI dashboard aesthetic., render_about_section(), render_header(), render_sidebar()

### Community 1 - "random_forest_hr/app/streamlit_app.py"
Cohesion: 0.21
Nodes (8): cache_data, employee_dataframe(), load_csv(), load_metadata(), load_pipeline(), probability_chart(), cache_resource, DataFrame

### Community 2 - "regression/src/preprocessing.py"
Cohesion: 0.24
Nodes (10): align_features(), build_feature_matrix(), cap_outliers_iqr(), clean_raw_dataframe(), DataFrame, Reusable preprocessing utilities for the loan amount regression project. These…, Apply structural cleaning to the raw loan approval dataset. Strips whitespace…, Apply Interquartile Range based winsorization to the specified numerical… (+2 more)

### Community 3 - "Path"
Cohesion: 0.20
Nodes (7): Path, load_artifacts(), cache_resource, load_pipeline(), cache_resource, load_pipeline(), cache_resource

### Community 4 - "nti_ml_app/utils/model_loader.py"
Cohesion: 0.25
Nodes (7): get_model_path(), get_project_root(), load_artifact(), cache_resource, Loads a .pkl or .joblib file given a path relative to the repository root., A helper mapping model names to their relative paths., Returns the path to the NTI-ML repository root.

### Community 5 - "build_preprocessor"
Cohesion: 0.50
Nodes (4): ColumnTransformer, build_preprocessor(), clean_hr_data(), DataFrame

## Knowledge Gaps
- **222 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_csv()` connect `random_forest_hr/app/streamlit_app.py` to `Path`?**
  _High betweenness centrality (0.003) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Path` (e.g. with `load_artifacts()` and `load_pipeline()`) actually correct?**
  _`Path` has 3 INFERRED edges - model-reasoned connections that need verification._