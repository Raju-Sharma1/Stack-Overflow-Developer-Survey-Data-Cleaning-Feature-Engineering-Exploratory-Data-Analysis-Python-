import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Stack Overflow survey dataset directly from hosted cloud storage
data_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/n01PQ9pSmiRX6520flujwQ/survey-data.csv'
df = pd.read_csv(data_url)

# Quick preview to understand structure and columns
df.head()


# ================================
# Handling Missing Data (Critical Columns)
# ================================

# Inspect unique values to understand categorical structure
df["Employment"].unique()
df["JobSat"].unique()
df["RemoteWork"].unique()


# ---- Employment Column ----
# Checking whether Employment column contains missing values
print("Checking 'Employment' column for missing values...")
if df["Employment"].isnull().sum() > 0:
    print("Found missing values:", df["Employment"].isnull().sum())
else:
    print("No missing values found in 'Employment' column")

print("="*60)
print("Moving to 'JobSat' Column")
print("="*60)


# ---- Job Satisfaction Column ----
# Imputing missing values using median (robust to outliers in ordinal scale)
print("Checking 'JobSat' column for missing values...")
if df["JobSat"].isnull().sum() > 0:
    jobsat_median = df["JobSat"].median()
    df["JobSat"] = df["JobSat"].fillna(jobsat_median)

    if df["JobSat"].isnull().sum() == 0:
        print("Missing values successfully imputed using median.")
else:
    print("No missing values found in 'JobSat' column")

print("="*60)
print("Moving to 'RemoteWork' Column")
print("="*60)


# ---- RemoteWork Column ----
# Using mode imputation since this is a categorical variable
print("Checking 'RemoteWork' column for missing values...")
if df["RemoteWork"].isnull().sum() > 0:
    most_frequent_value = df["RemoteWork"].mode()[0]
    df["RemoteWork"] = df["RemoteWork"].fillna(most_frequent_value)

    if df["RemoteWork"].isnull().sum() == 0:
        print("Missing values successfully imputed using mode.")
else:
    print("No missing values found in 'RemoteWork' column")

print("Missing data handled for Employment, JobSat, and RemoteWork columns.")


# ================================
# Experience & Job Satisfaction Analysis
# ================================

import numpy as np

# Converting textual experience values into numeric approximations
df["YearsCodePro"] = df["YearsCodePro"].replace({
    "Less than 1 year": 0.5,
    "More than 50 years": 50
})

# Filling missing experience values using median to maintain distribution balance
df["YearsCodePro"] = df["YearsCodePro"].fillna(df["YearsCodePro"].median())

# Ensuring numeric datatype for further quantitative analysis
df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"], errors='coerce')

# Creating experience buckets for grouped analysis
bins = [0, 5, 10, 20, np.inf]
labels = ["0-5", "5-10", "10-20", ">20"]
df["YearsCodeProLevel"] = pd.cut(df["YearsCodePro"], bins=bins, labels=labels, right=False)

# Aggregating median Job Satisfaction across experience groups
median_jobsat = df.groupby("YearsCodeProLevel")["JobSat"].median()

# Visualizing experience vs satisfaction trend
median_jobsat.plot(kind="bar")
plt.title("Median Job Satisfaction by Experience Level")
plt.ylabel("Median Job Satisfaction")
plt.show()


# ================================
# Distribution Analysis
# ================================

# Job satisfaction frequency distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="JobSat", data=df)
plt.title("Distribution of Job Satisfaction (JobSat)")
plt.show()

# Remote work preference distribution
plt.figure(figsize=(12,4))
sns.countplot(x="RemoteWork", data=df)
plt.title("Distribution of RemoteWork Preferences")
plt.grid(axis="y", linestyle="--")
plt.show()


# ================================
# Programming Language Trends by Region
# ================================

# Replace missing categorical values to prevent grouping errors
df["Country"] = df["Country"].fillna("Unknown")
df["LanguageHaveWorkedWith"] = df["LanguageHaveWorkedWith"].fillna("Not Specified")

# Split multi-language entries into individual rows for accurate counting
df["LanguageHaveWorkedWith"] = df["LanguageHaveWorkedWith"].str.split(";")
df = df.explode("LanguageHaveWorkedWith")

# Identify top 5 countries by respondent count
top_countries = df["Country"].value_counts().head(5).index
df_top = df[df["Country"].isin(top_countries)]

# Count language usage per country
lang_region_count = (
    df_top
    .groupby(["Country", "LanguageHaveWorkedWith"])
    .size()
    .reset_index(name="Count")
)

# Select top 5 languages per country
top_langs = (
    lang_region_count
    .sort_values(["Country", "Count"], ascending=[True, False])
    .groupby("Country")
    .head(5)
)

# Visual comparison across regions
plt.figure(figsize=(14,6))
sns.barplot(
    data=top_langs,
    x="LanguageHaveWorkedWith",
    y="Count",
    hue="Country"
)
plt.xticks(rotation=45)
plt.title("Top Programming Languages by Region")
plt.show()


# ================================
# Experience vs Satisfaction Correlation
# ================================

# Convert satisfaction points column to numeric for scatter analysis
df["JobSatPoints_1"] = pd.to_numeric(df["JobSatPoints_1"], errors="coerce")

# Remove incomplete rows before plotting
scatter_df = df.dropna(subset=["YearsCodePro", "JobSatPoints_1"])

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=scatter_df,
    x="YearsCodePro",
    y="JobSatPoints_1",
    alpha=0.5
)
plt.title("Relationship Between Experience and Job Satisfaction")
plt.show()


# ================================
# Education vs Employment Analysis
# ================================

# Expanding multi-employment responses for accurate cross-tabulation
ed_emp = df[["EdLevel", "Employment"]].copy()
ed_emp["Employment"] = ed_emp["Employment"].str.split(";")
ed_emp = ed_emp.explode("Employment")

# Normalized cross-tab to compare proportional distribution
edu_emp_crosstab = pd.crosstab(
    ed_emp["EdLevel"],
    ed_emp["Employment"],
    normalize="index"
)

# Stacked visualization for comparative insight
edu_emp_crosstab.plot(
    kind="bar",
    stacked=True,
    figsize=(12,6)
)

plt.title("Education Level vs Employment Type")
plt.xticks(rotation=45)
plt.legend(title="Employment Type", bbox_to_anchor=(1.05, 1))
plt.show()
