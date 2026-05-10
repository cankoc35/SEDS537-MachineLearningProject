📢 SEDS 537 Term Project – Progress Report Requirements
Dear Students, SEDS537_Fall2025
 
Next week, you are expected to submit a Progress Report for your SEDS 537 – Machine Learning Individual Term Project.
The submission deadline is Monday, May 11, 2026, at 23:59.
 
Submission through the designated folder on the Microsoft Teams course page is sufficient. You do not need to send the report by email.
 
This report corresponds to the Technical Checkpoint stage in the project timeline. The purpose is to check whether your project has moved beyond the proposal stage and whether you have made sufficient progress in dataset preparation, literature review, baseline implementation, initial experiments, and reproducible development.
 
At this stage, your project does not need to be completed. However, your report should clearly show that your work is technically feasible, organized, and on track for the final submission.
Expected Content of the Progress Report
Your report should include the following sections.
1. Project Title and Problem Definition
State your project title and briefly explain the problem you are addressing.
You should clearly describe:
What machine learning task you are working on;
Why the problem is important;
What dataset you are using;
What the input and output of the system are;
What makes the problem challenging.
If your topic or title has changed since the proposal, briefly explain the change.
2. Dataset and Preprocessing Status
Describe the dataset you are using and your current progress with it.
Please include:
Dataset name and source;
Number of samples;
Features or input type;
Target labels or output, if applicable;
Train/validation/test split;
Preprocessing steps completed;
Any problems such as missing values, class imbalance, noisy data, or access issues.
You should clearly state whether the dataset has already been downloaded, cleaned, and prepared for experiments.
3. Literature Review Progress
Briefly summarize your literature review progress.
You should mention:
The main papers, methods, or models you have reviewed so far;
How they are related to your project;
Which methods are commonly used for this problem;
Which methods may be used as baselines;
What technical direction your project is likely to follow.
A simple list of paper titles is not enough. Please briefly explain how the reviewed works support your project.
4. Baseline Models
Each project must compare at least three baseline models in the final submission.
In this section, report the current status of your baselines:
Which baseline models you selected;
Why they are suitable for your project;
Which ones have already been implemented;
Whether you have obtained any initial results;
Any implementation problems you encountered.
At this checkpoint, you are expected to have started implementing the baseline models.
5. Initial Experimental Results
Include any preliminary experimental results you have obtained so far.
Depending on your project, you may report metrics such as:
Accuracy;
Precision;
Recall;
F1-score;
Macro-F1;
AUROC;
RMSE;
MAE;
NDCG;
Silhouette score;
Calibration error;
Runtime or memory usage.
Please include a small table if you already have results.
Example:
Model	Metric 1	Metric 2	Notes
Baseline 1	-	-	Implemented / In progress
Baseline 2	-	-	Implemented / In progress
Baseline 3	-	-	Implemented / In progress
Do not only report numbers. Briefly interpret what the initial results indicate.
6. Planned Improvements and Technical Direction
Instead of requiring a finalized proposed method at this stage, you should briefly explain the technical direction you plan to follow in the next phase of the project.
This may include improving the best-performing baseline, adding feature engineering, trying a stronger model, addressing class imbalance, improving preprocessing, using hyperparameter optimization, adding regularization, improving interpretability, testing robustness, or comparing alternative representations.
This section does not need to present a complete new method. It should only explain how you plan to improve or extend the current experimental pipeline.
7. Ablation and Error Analysis Plan
The final project must include both ablation study and error analysis. In the progress report, you should briefly explain your plan for these analyses.
For ablation study, explain which parts of your pipeline you may compare or remove to understand their contribution.
For error analysis, explain how you plan to examine model failures, such as misclassified examples, confusion matrix patterns, minority-class errors, difficult samples, overconfident wrong predictions, or failure cases of baseline models.
You do not need to complete these analyses yet, but you should have a clear plan.
8. Visualization and Interpretation
Mention the visualizations or interpretation methods you plan to use.
Examples include confusion matrices, learning curves, ROC or precision-recall curves, feature importance plots, t-SNE or UMAP visualizations, attention maps, calibration plots, or error distribution plots.
Visualizations should help explain the results, not only decorate the report.
9. GitHub and Reproducibility Status
Your project must be reproducible through GitHub.
Please include:
GitHub repository link;
Current repository structure;
Implemented scripts;
README status;
Dependency file status, such as requirements.txt or environment.yml;
Instructions for running the current experiments;
Dataset preparation instructions;
Commit history status.
Please remember that project progress will also be monitored through GitHub commits. Consistent development is expected.
10. Current Challenges and Next Steps
Briefly explain your current challenges and your plan for the remaining weeks.
You may discuss dataset problems, implementation difficulties, low performance, computational limitations, model training problems, or evaluation difficulties.
Then summarize what you plan to complete before the final submission.
Suggested Report Format
The progress report should be written in a clear and organized academic style.
Recommended length: 3–5 pages
Suggested structure:
Project Title and Problem Definition
Dataset and Preprocessing Status
Literature Review Progress
Baseline Models
Initial Experimental Results
Planned Improvements and Technical Direction
Ablation and Error Analysis Plan
Visualization Plan
GitHub and Reproducibility Status
Current Challenges and Next Steps
Submission Deadline and Method
The progress report must be submitted by:
Monday, May 11, 2026, 23:59
 
Please upload your report to the designated folder on the Microsoft Teams course page. Submission through Teams is sufficient; you do not need to send the report by email.
 
Late submissions may negatively affect the project evaluation.


"
MY NOTES FOR YOU CODEX
for the next Improvements, mention that i might add a different LLM model for the extracting scores from the hauleval, maybe a smallar LLM, to test if the results will change.

the current classification scores are very high. it might good great but we will test it with truthfulqa dataset, which is a more challenging dataset for LLMs. This will help us understand if the high scores are due to overfitting on the current dataset or if the model truly generalizes well.

github link:
https://github.com/cankoc35/SEDS537-MachineLearningProject
"
 