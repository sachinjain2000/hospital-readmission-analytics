"""
Hospital Readmission Analytics - Data Exploration and Preprocessing
Author: Data Analytics Portfolio Project
Date: February 2026

This script performs comprehensive data exploration and preprocessing on the
diabetes hospital readmission dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# File paths
DATA_PATH = '../data/diabetic_data.csv'
OUTPUT_PATH = '../visualizations/'

def load_data():
    """Load the dataset and display basic information"""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def explore_basic_info(df):
    """Display basic dataset information"""
    print("\n" + "=" * 80)
    print("BASIC DATASET INFORMATION")
    print("=" * 80)
    
    print("\nColumn Names and Data Types:")
    print(df.dtypes)
    
    print("\nFirst 5 Rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    print("\n" + "=" * 80)
    print("MISSING VALUE ANALYSIS")
    print("=" * 80)
    
    # Calculate missing values
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Question_Mark_Count': (df == '?').sum(),
        'Question_Mark_Percentage': ((df == '?').sum() / len(df)) * 100
    })
    
    missing['Total_Missing'] = missing['Missing_Count'] + missing['Question_Mark_Count']
    missing['Total_Percentage'] = missing['Missing_Percentage'] + missing['Question_Mark_Percentage']
    missing = missing.sort_values('Total_Percentage', ascending=False)
    
    print("\nMissing Values Summary:")
    print(missing[missing['Total_Percentage'] > 0])
    
    # Visualize missing values
    fig, ax = plt.subplots(figsize=(12, 8))
    top_missing = missing[missing['Total_Percentage'] > 5].sort_values('Total_Percentage')
    
    if len(top_missing) > 0:
        ax.barh(top_missing['Column'], top_missing['Total_Percentage'], color='coral')
        ax.set_xlabel('Missing Percentage (%)', fontsize=12)
        ax.set_ylabel('Column Name', fontsize=12)
        ax.set_title('Columns with >5% Missing Values', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + '01_missing_values.png', dpi=300, bbox_inches='tight')
        print(f"\nMissing values visualization saved to: {OUTPUT_PATH}01_missing_values.png")
    
    return missing

def analyze_target_variable(df):
    """Analyze the target variable: readmitted"""
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS: READMITTED")
    print("=" * 80)
    
    print("\nReadmission Distribution:")
    print(df['readmitted'].value_counts())
    print("\nReadmission Percentages:")
    print(df['readmitted'].value_counts(normalize=True) * 100)
    
    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    readmit_counts = df['readmitted'].value_counts()
    axes[0].bar(readmit_counts.index, readmit_counts.values, color=['#2ecc71', '#e74c3c', '#f39c12'])
    axes[0].set_xlabel('Readmission Status', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Readmission Status', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(readmit_counts.values):
        axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    axes[1].pie(readmit_counts.values, labels=readmit_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Readmission Rate Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + '02_target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nTarget distribution visualization saved to: {OUTPUT_PATH}02_target_distribution.png")

def analyze_demographics(df):
    """Analyze demographic features"""
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC ANALYSIS")
    print("=" * 80)
    
    # Age distribution
    print("\nAge Distribution:")
    print(df['age'].value_counts().sort_index())
    
    # Gender distribution
    print("\nGender Distribution:")
    print(df['gender'].value_counts())
    
    # Race distribution
    print("\nRace Distribution:")
    print(df['race'].value_counts())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Age distribution
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_counts = df['age'].value_counts().reindex(age_order, fill_value=0)
    axes[0, 0].bar(range(len(age_counts)), age_counts.values, color='steelblue')
    axes[0, 0].set_xticks(range(len(age_counts)))
    axes[0, 0].set_xticklabels(age_counts.index, rotation=45, ha='right')
    axes[0, 0].set_xlabel('Age Group', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Patient Age Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Gender distribution
    gender_counts = df['gender'].value_counts()
    axes[0, 1].bar(gender_counts.index, gender_counts.values, color=['#3498db', '#e91e63'])
    axes[0, 1].set_xlabel('Gender', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Patient Gender Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Race distribution
    race_counts = df['race'].value_counts()
    axes[1, 0].barh(race_counts.index, race_counts.values, color='coral')
    axes[1, 0].set_xlabel('Count', fontsize=11)
    axes[1, 0].set_ylabel('Race', fontsize=11)
    axes[1, 0].set_title('Patient Race Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Time in hospital
    time_stats = df['time_in_hospital'].describe()
    axes[1, 1].hist(df['time_in_hospital'], bins=14, color='teal', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(time_stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {time_stats["mean"]:.1f}')
    axes[1, 1].axvline(time_stats['50%'], color='orange', linestyle='dashed', linewidth=2, label=f'Median: {time_stats["50%"]:.1f}')
    axes[1, 1].set_xlabel('Days in Hospital', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Length of Hospital Stay Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + '03_demographics.png', dpi=300, bbox_inches='tight')
    print(f"\nDemographic visualizations saved to: {OUTPUT_PATH}03_demographics.png")

def analyze_medical_features(df):
    """Analyze medical features"""
    print("\n" + "=" * 80)
    print("MEDICAL FEATURES ANALYSIS")
    print("=" * 80)
    
    # Key medical features
    medical_features = ['num_lab_procedures', 'num_procedures', 'num_medications', 
                        'number_outpatient', 'number_emergency', 'number_inpatient']
    
    print("\nMedical Features Statistics:")
    print(df[medical_features].describe())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(medical_features):
        axes[idx].hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        mean_val = df[feature].mean()
        axes[idx].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                          label=f'Mean: {mean_val:.1f}')
        axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'{feature.replace("_", " ").title()} Distribution', 
                           fontsize=11, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + '04_medical_features.png', dpi=300, bbox_inches='tight')
    print(f"\nMedical features visualizations saved to: {OUTPUT_PATH}04_medical_features.png")

def create_summary_report(df, missing):
    """Create a summary report of the exploration"""
    print("\n" + "=" * 80)
    print("EXPLORATION SUMMARY REPORT")
    print("=" * 80)
    
    summary = {
        'Total Records': f"{len(df):,}",
        'Total Features': df.shape[1],
        'Duplicate Rows': df.duplicated().sum(),
        'Columns with Missing Values': len(missing[missing['Total_Percentage'] > 0]),
        'Numeric Features': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical Features': len(df.select_dtypes(include=['object']).columns),
        'Readmission Rate (<30 days)': f"{(df['readmitted'] == '<30').sum() / len(df) * 100:.2f}%",
        'Average Hospital Stay': f"{df['time_in_hospital'].mean():.2f} days",
        'Average Age Group': df['age'].mode()[0],
        'Most Common Gender': df['gender'].mode()[0],
    }
    
    print("\nKey Statistics:")
    for key, value in summary.items():
        print(f"  • {key}: {value}")
    
    return summary

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("HOSPITAL READMISSION ANALYTICS - DATA EXPLORATION")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Basic exploration
    df = explore_basic_info(df)
    
    # Missing value analysis
    missing = analyze_missing_values(df)
    
    # Target variable analysis
    analyze_target_variable(df)
    
    # Demographic analysis
    analyze_demographics(df)
    
    # Medical features analysis
    analyze_medical_features(df)
    
    # Summary report
    summary = create_summary_report(df, missing)
    
    print("\n" + "=" * 80)
    print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Data Cleaning and Preprocessing")
    print("  2. Feature Engineering")
    print("  3. Predictive Modeling")
    print("  4. Dashboard Creation")

if __name__ == "__main__":
    main()
