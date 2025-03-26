# -*- coding: utf-8 -*-
"""Omega Project - AI_Cancer_Detection_Breast_Cancer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1A2MWKpTDT583vXubxbrFP5bMc0wJiGCe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Upload Breast_Cancer.csv dataset
from google.colab import files
dataset = files.upload()

df = pd.read_csv('Breast_Cancer.csv')

"""Cleaning dataset"""

# missing values
print(df.isnull().sum())

"""No missing/null values"""

# Convert columns with categorical values to numerical values

# Status
df["Status"] = df["Status"].replace({"Alive":1, "Dead":2})

# Progesterone Status
df["Progesterone Status"] = df["Progesterone Status"].replace({"Positive":1, "Negative":2})

# Estrogen Status
df["Estrogen Status"] = df["Estrogen Status"].replace({"Positive":1, "Negative":2})

# A Stage
df["A Stage"] = df["A Stage"].replace({"Localized":1, "Regional":2, "Distant":3})

# Grade
df["Grade"] = df["Grade"].replace(" anaplastic; Grade IV", 4)
df["Grade"] = pd.to_numeric(df["Grade"])

# Marital status
df["Marital Status"] = df["Marital Status"].replace({"Married":1, "Divorced":2, "Widowed":3, "Single ":4, "Separated":5})

# Differentiate
df["differentiate"] = df["differentiate"].replace({"Well differentiated":1, "Moderately differentiated":2, "Poorly differentiated":3, "Undifferentiated":4})

# Race
df["Race"] = df["Race"].replace({"White":1, "Black":2, "Other":3})

# T Stage
df["T Stage "] = df["T Stage "].replace({"T1":1, "T2":2, "T3":3, "T4":4})

# N Stage
df["N Stage"] = df["N Stage"].replace({"N0":1, "N1":2, "N2":3, "N3":4})

# 6th Stage
df["6th Stage"] = df["6th Stage"].replace({"I":1, "IIA":2, "IIB":2, "IIIA":3, "IIIB":3, "IIIC":3})

pd.set_option('future.no_silent_downcasting', True)

# see info
df.info()
df.head()

"""Converted categorical to numerical data"""

# Download cleaned dataset
df.to_csv("cleaned_breast_cancer_data.csv", index=False)
files.download("cleaned_breast_cancer_data.csv")

"""Data analysis queries below

1. Analyze survival months and survival status (last 2 columns)
"""

# Relationship between tumour size and survival months
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Tumor Size"], y=df["Survival Months"], color="teal")

plt.xlabel("Tumor Size")
plt.ylabel("Survival Months")
plt.title("Relationship between Tumor Size and Survival Months")
plt.show()

# Relationship between tumour size and survival status
plt.figure(figsize=(8, 5))
sns.barplot(x=df["Status"], y=df["Tumor Size"], palette="coolwarm", hue=df["Status"], legend=False)

plt.xlabel("Survival Status (1 = Alive, 2 = Deceased)")
plt.ylabel("Average Tumor Size")
plt.title("Average Tumor Size vs Survival Status")
plt.show()

# Relationship between tumour aggressiveness (grade) and survival months
plt.figure(figsize=(8, 5))
sns.barplot(x=df["Grade"], y=df["Survival Months"], palette="viridis", hue=df["Grade"], legend=False)

plt.xlabel("Tumor Grade")
plt.ylabel("Survival Months")
plt.title("Survival Months vs Tumor Grade")
plt.show()

# Relationship between tumour aggressiveness (grade) and survival status
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Grade"], hue=df["Status"], palette="coolwarm")

plt.xlabel("Tumor Grade")
plt.ylabel("Count")
plt.title("Survival Status vs Tumor Grade")
plt.show()

# Relationship between age and survival months
plt.figure(figsize=(8, 5))
sns.lineplot(x=df["Age"], y=df["Survival Months"], color="blue")

plt.xlabel("Age")
plt.ylabel("Survival Months")
plt.title("Survival Months vs Age")
plt.show()

# Relationship between N stage and survival months
avg_survival_months = df.groupby("N Stage")["Survival Months"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_survival_months.index, y=avg_survival_months.values, palette="coolwarm", hue=avg_survival_months.index, legend=False)

# Set labels and title
plt.xlabel("N Stage")
plt.ylabel("Average Survival Months")
plt.title("Average Survival Months vs N Stage")
plt.show()

# Relationship between N stage and survival rate
plt.figure(figsize=(8, 5))
sns.countplot(x=df["N Stage"], hue=df["Status"], palette="coolwarm")

plt.xlabel("N Stage")
plt.ylabel("Count")
plt.title("Survival Status vs N Stage")
plt.legend(title="Survival Status", labels=["Deceased", "Alive"])
plt.show()

"""2. Other analyses"""

# Age distribution of breast cancer patients
df["Age"].hist(bins=20)

# Race distribution
colors = ["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6"]

race_lbl_map = {
    1: "White",
    2: "Black",
    3: "Other"
}

# Replace values only for visualization
race_counts = df["Race"].value_counts()
race_labels = [race_lbl_map.get(race, "Unknown") for race in race_counts.index]

# Define custom colors
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

plt.figure(figsize=(6, 6))
plt.pie(
    race_counts,
    labels=race_labels,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops={"edgecolor": "black"}
)

plt.title("Race Distribution in the Dataset")
plt.ylabel("")  # Hide the y-label
plt.show()

# Marital status vs survival months
df.groupby("Marital Status")["Survival Months"].mean()

"""No effect."""

# Tumour size with top ten most common sizes
tumor_counts = df["Tumor Size"].value_counts().head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=tumor_counts.index, y=tumor_counts.values, palette="magma", hue=tumor_counts.index, legend=False)

plt.xlabel("Tumor Size")
plt.ylabel("Count")
plt.title("Top 10 Most Common Tumor Sizes")
plt.xticks(rotation=45)  # Rotate labels if needed
plt.show()

# Survival rate and 6th stage
df.groupby("6th Stage")["Survival Months"].mean()

# Hormone receptors and survival rate
df.groupby("Estrogen Status")["Survival Months"].mean()

df.groupby("Progesterone Status")["Survival Months"].mean()

"""Weak positive correlation between hormone receptors and survival rate?"""