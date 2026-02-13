"""
Titanic Survival Prediction - Complete Analysis
================================================
This script performs:
1. Data Exploration & Visualization
2. Data Cleaning (handling missing values)
3. ML Model Training & Comparison
4. Final Survivor List Generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("🚢 TITANIC SURVIVAL PREDICTION PROJECT")
print("=" * 60)

# ============================================================
# PHASE 1: DATA LOADING & EXPLORATION
# ============================================================
print("\n📂 PHASE 1: Loading and Exploring Data...")
print("-" * 60)

# Load the dataset
df = pd.read_csv('titanic.csv')
df_original = df.copy()  # Keep original for final output

print(f"✅ Dataset loaded successfully!")
print(f"📊 Total Passengers: {len(df)}")
print(f"📊 Total Features: {len(df.columns)}")

# Display basic info
print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

# Missing values analysis
print("\n❌ Missing Values Analysis:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ============================================================
# PHASE 2: COMPREHENSIVE VISUALIZATIONS
# ============================================================
print("\n📊 PHASE 2: Creating Visualizations...")
print("-" * 60)

# Create output directory for charts
import os
charts_dir = 'charts'
os.makedirs(charts_dir, exist_ok=True)

# Chart 1: Survival Rate Pie Chart
print("📈 Creating Chart 1: Survival Rate Pie Chart...")
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#ff6b6b', '#4ecdc4']
survival_counts = df['Survived'].value_counts()
labels = ['Did Not Survive', 'Survived']
explode = (0.05, 0.05)
ax.pie(survival_counts, labels=labels, autopct='%1.1f%%', colors=colors, 
       explode=explode, shadow=True, startangle=90, textprops={'fontsize': 14})
ax.set_title('Overall Survival Rate on Titanic', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{charts_dir}/01_survival_pie_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Survival by Gender
print("📈 Creating Chart 2: Survival by Gender...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
survival_gender = df.groupby(['Sex', 'Survived']).size().unstack()
survival_gender.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
axes[0].set_title('Survival Count by Gender', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Gender', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')
axes[0].tick_params(axis='x', rotation=0)

# Survival rate
survival_rate_gender = df.groupby('Sex')['Survived'].mean() * 100
survival_rate_gender.plot(kind='bar', ax=axes[1], color=['#ff9f43', '#a55eea'], edgecolor='black')
axes[1].set_title('Survival Rate by Gender (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Gender', fontsize=12)
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(survival_rate_gender):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/02_survival_by_gender.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Survival by Passenger Class
print("📈 Creating Chart 3: Survival by Passenger Class...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

survival_class = df.groupby(['Pclass', 'Survived']).size().unstack()
survival_class.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
axes[0].set_title('Survival Count by Passenger Class', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Passenger Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].legend(['Did Not Survive', 'Survived'])
axes[0].tick_params(axis='x', rotation=0)

survival_rate_class = df.groupby('Pclass')['Survived'].mean() * 100
survival_rate_class.plot(kind='bar', ax=axes[1], color=['#ffd93d', '#6bcb77', '#4d96ff'], edgecolor='black')
axes[1].set_title('Survival Rate by Passenger Class (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Passenger Class', fontsize=12)
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(survival_rate_class):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/03_survival_by_class.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Age Distribution
print("📈 Creating Chart 4: Age Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
df['Age'].dropna().hist(bins=30, ax=axes[0], color='#74b9ff', edgecolor='black', alpha=0.8)
axes[0].set_title('Age Distribution of Passengers', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# KDE by survival
df[df['Survived'] == 0]['Age'].dropna().plot.kde(ax=axes[1], label='Did Not Survive', color='#ff6b6b', linewidth=2)
df[df['Survived'] == 1]['Age'].dropna().plot.kde(ax=axes[1], label='Survived', color='#4ecdc4', linewidth=2)
axes[1].set_title('Age Distribution by Survival', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Age', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{charts_dir}/04_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Survival by Age Group
print("📈 Creating Chart 5: Survival by Age Group...")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 80], labels=['Child (0-12)', 'Teen (13-18)', 'Adult (19-35)', 'Middle (36-50)', 'Senior (51+)'])
fig, ax = plt.subplots(figsize=(12, 6))
survival_age = df.groupby('AgeGroup')['Survived'].mean() * 100
survival_age.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#a55eea'], edgecolor='black')
ax.set_title('Survival Rate by Age Group (%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Survival Rate (%)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(survival_age):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{charts_dir}/05_survival_by_age_group.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Fare Distribution
print("📈 Creating Chart 6: Fare Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df['Fare'].hist(bins=50, ax=axes[0], color='#fdcb6e', edgecolor='black', alpha=0.8)
axes[0].set_title('Fare Distribution', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Fare', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

sns.boxplot(x='Pclass', y='Fare', data=df, ax=axes[1], palette='Set2')
axes[1].set_title('Fare by Passenger Class', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Passenger Class', fontsize=12)
axes[1].set_ylabel('Fare', fontsize=12)

plt.tight_layout()
plt.savefig(f'{charts_dir}/06_fare_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 7: Survival by Embarked Port
print("📈 Creating Chart 7: Survival by Embarked Port...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

embarked_labels = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
df['EmbarkedName'] = df['Embarked'].map(embarked_labels)

survival_embarked = df.groupby(['EmbarkedName', 'Survived']).size().unstack()
survival_embarked.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
axes[0].set_title('Survival Count by Embarked Port', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Port', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].legend(['Did Not Survive', 'Survived'])
axes[0].tick_params(axis='x', rotation=45)

survival_rate_embarked = df.groupby('EmbarkedName')['Survived'].mean() * 100
survival_rate_embarked.plot(kind='bar', ax=axes[1], color=['#e17055', '#00b894', '#0984e3'], edgecolor='black')
axes[1].set_title('Survival Rate by Embarked Port (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Port', fontsize=12)
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(survival_rate_embarked):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/07_survival_by_embarked.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 8: Family Size Analysis
print("📈 Creating Chart 8: Family Size Analysis...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

survival_family = df.groupby('FamilySize')['Survived'].mean() * 100
survival_family.plot(kind='bar', ax=axes[0], color='#74b9ff', edgecolor='black')
axes[0].set_title('Survival Rate by Family Size (%)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Family Size', fontsize=12)
axes[0].set_ylabel('Survival Rate (%)', fontsize=12)
axes[0].tick_params(axis='x', rotation=0)

alone_labels = ['With Family', 'Alone']
survival_alone = df.groupby('IsAlone')['Survived'].mean() * 100
survival_alone.index = alone_labels
survival_alone.plot(kind='bar', ax=axes[1], color=['#00b894', '#e17055'], edgecolor='black')
axes[1].set_title('Survival Rate: Alone vs With Family (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Status', fontsize=12)
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(survival_alone):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/08_family_size_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 9: Title Analysis
print("📈 Creating Chart 9: Title Analysis...")
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Simplify titles
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

title_counts = df['Title'].value_counts()
title_counts.plot(kind='bar', ax=axes[0], color=['#0984e3', '#e17055', '#00b894', '#fdcb6e', '#a55eea'], edgecolor='black')
axes[0].set_title('Passenger Count by Title', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Title', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

survival_title = df.groupby('Title')['Survived'].mean() * 100
survival_title.plot(kind='bar', ax=axes[1], color=['#0984e3', '#e17055', '#00b894', '#fdcb6e', '#a55eea'], edgecolor='black')
axes[1].set_title('Survival Rate by Title (%)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Title', fontsize=12)
axes[1].set_ylabel('Survival Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(survival_title):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/09_title_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 10: Correlation Heatmap
print("📈 Creating Chart 10: Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(12, 10))
# Select numeric columns for correlation
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', 
            linewidths=0.5, ax=ax, annot_kws={'size': 12})
ax.set_title('Feature Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{charts_dir}/10_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 11: Combined Gender & Class Analysis
print("📈 Creating Chart 11: Gender & Class Combined...")
fig, ax = plt.subplots(figsize=(12, 6))
survival_gender_class = df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
survival_gender_class.unstack().plot(kind='bar', ax=ax, color=['#e17055', '#0984e3'], edgecolor='black')
ax.set_title('Survival Rate by Class and Gender (%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Passenger Class', fontsize=12)
ax.set_ylabel('Survival Rate (%)', fontsize=12)
ax.legend(['Female', 'Male'])
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{charts_dir}/11_gender_class_combined.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ All 11 charts created and saved to 'charts' folder!")

# ============================================================
# PHASE 3: DATA CLEANING & FEATURE ENGINEERING
# ============================================================
print("\n🧹 PHASE 3: Data Cleaning & Feature Engineering...")
print("-" * 60)

# Handle missing Age values using median by Title
print("🔧 Filling missing Age values using Title-based median...")
age_by_title = df.groupby('Title')['Age'].median()
for title in df['Title'].unique():
    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_by_title[title]

# Fill remaining missing ages with overall median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing Embarked values (fill with mode)
print("🔧 Filling missing Embarked values with mode...")
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Handle missing Fare values
print("🔧 Filling missing Fare values with median...")
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop Cabin column (too many missing values)
print("🔧 Dropping Cabin column (77% missing)...")
df.drop('Cabin', axis=1, inplace=True)

# Encode categorical variables
print("🔧 Encoding categorical variables...")
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()

df['Sex_Encoded'] = le_sex.fit_transform(df['Sex'])
df['Embarked_Encoded'] = le_embarked.fit_transform(df['Embarked'])
df['Title_Encoded'] = le_title.fit_transform(df['Title'])

# Verify no missing values
print("\n✅ Missing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values")

# ============================================================
# PHASE 4: ML MODEL BUILDING & COMPARISON
# ============================================================
print("\n🤖 PHASE 4: Machine Learning Model Building...")
print("-" * 60)

# Select features for model
features = ['Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
            'Embarked_Encoded', 'FamilySize', 'IsAlone', 'Title_Encoded']
X = df[features]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set size: {len(X_train)}")
print(f"📊 Test set size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
print("\n📈 Training and Evaluating Models...")
print("-" * 50)
results = {}

for name, model in models.items():
    # Train
    if name in ['Support Vector Machine', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X if name not in ['Support Vector Machine', 'Logistic Regression'] else scaler.fit_transform(X), y, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n🔹 {name}:")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Cross-Val Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"🎯 ACCURACY: {best_accuracy * 100:.2f}%")
print("=" * 60)

# Chart 12: Model Comparison
print("\n📈 Creating Chart 12: Model Comparison...")
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#a55eea']
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax.set_title('Model Accuracy Comparison', fontsize=18, fontweight='bold')
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_ylim(0, 100)
ax.axhline(y=best_accuracy*100, color='red', linestyle='--', linewidth=2, label=f'Best: {best_accuracy*100:.1f}%')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', 
            ha='center', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{charts_dir}/12_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# PHASE 5: GENERATE FINAL SURVIVOR LIST
# ============================================================
print("\n📋 PHASE 5: Generating Final Survivor List...")
print("-" * 60)

# Use the best model to predict on all data
if best_model_name in ['Support Vector Machine', 'Logistic Regression']:
    best_model = models[best_model_name]
    X_all_scaled = scaler.fit_transform(X)
    df['Predicted_Survival'] = best_model.predict(X_all_scaled)
else:
    best_model = models[best_model_name]
    df['Predicted_Survival'] = best_model.predict(X)

# Create survivor list
survivors = df[df['Survived'] == 1][['PassengerId', 'Name', 'Age', 'Sex', 'Pclass']].copy()
survivors = survivors.sort_values('PassengerId')

print(f"\n✅ Total Survivors: {len(survivors)}")
print(f"   - Males: {len(survivors[survivors['Sex'] == 'male'])}")
print(f"   - Females: {len(survivors[survivors['Sex'] == 'female'])}")
print(f"   - Average Age: {survivors['Age'].mean():.1f} years")

# Save survivor list to CSV
survivors.to_csv('survivors_list.csv', index=False)
print("\n📁 Survivor list saved to 'survivors_list.csv'")

# Display survivor list
print("\n" + "=" * 80)
print("📋 COMPLETE SURVIVOR LIST (Who Survived, Age, Gender)")
print("=" * 80)
print(survivors.to_string(index=False))

# Save a formatted survivor report
with open('survivor_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("🚢 TITANIC SURVIVOR REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Survivors: {len(survivors)}\n")
    f.write(f"Male Survivors: {len(survivors[survivors['Sex'] == 'male'])}\n")
    f.write(f"Female Survivors: {len(survivors[survivors['Sex'] == 'female'])}\n")
    f.write(f"Average Age of Survivors: {survivors['Age'].mean():.1f} years\n\n")
    f.write("-" * 80 + "\n")
    f.write("SURVIVOR LIST:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'ID':<6} {'Name':<55} {'Age':<8} {'Sex':<8}\n")
    f.write("-" * 80 + "\n")
    for _, row in survivors.iterrows():
        f.write(f"{row['PassengerId']:<6} {row['Name'][:54]:<55} {row['Age']:<8.1f} {row['Sex']:<8}\n")

print("\n📁 Detailed survivor report saved to 'survivor_report.txt'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated Files:")
print("   📊 charts/ - 12 visualization charts")
print("   📋 survivors_list.csv - Survivor data in CSV format")
print("   📝 survivor_report.txt - Detailed survivor report")
print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Model Accuracy: {best_accuracy * 100:.2f}%")
print("\n📊 Quick Statistics:")
print(f"   Total Passengers: {len(df)}")
print(f"   Survivors: {len(survivors)} ({len(survivors)/len(df)*100:.1f}%)")
print(f"   Did Not Survive: {len(df) - len(survivors)} ({(len(df) - len(survivors))/len(df)*100:.1f}%)")
print("\n" + "=" * 60)
