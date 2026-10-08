import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def load_modelling_data(db_path='loan_risk.db'):
    conn = sqlite3.connect(db_path)
    sql_query = """
        SELECT 
            b.borrower_id,
            b.age,
            b.income,
            b.employment_length,
            l.loan_id,
            l.loan_amount,
            l.loan_duration,
            l.loan_purpose,
            l.credit_history,
            l.existing_loans_count,
            r.default_flag
        FROM borrowers b
        JOIN loans l ON b.borrower_id = l.borrower_id
        JOIN repayment_status r ON l.loan_id = r.loan_id;
    """
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df


def engineer_features(df):
    df_engineered = df.copy()
    
    # ratio of total loan principal to annual income
    df_engineered['loan_to_income_ratio'] = df_engineered['loan_amount'] / df_engineered['income']
    
    # ratio of estimated monthly payment to estimated monthly income
    monthly_payment = df_engineered['loan_amount'] / df_engineered['loan_duration']
    monthly_income = df_engineered['income'] / 12.0
    df_engineered['installment_to_income_ratio'] = monthly_payment / monthly_income

    drop_cols = ['borrower_id', 'loan_id']
    df_engineered = df_engineered.drop(columns=drop_cols)
    
    categorical_cols = ['loan_purpose', 'credit_history']
    df_encoded = pd.get_dummies(df_engineered, columns=categorical_cols, drop_first=True)
    
    return df_encoded


def preprocess_and_split(df_encoded):
    X = df_encoded.drop(columns=['default_flag'])
    y = df_encoded['default_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    numerical_cols = [
        'age', 'income', 'employment_length', 'loan_amount',
        'loan_duration', 'existing_loans_count',
        'loan_to_income_ratio', 'installment_to_income_ratio'
    ]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_models(X_train, y_train):
    # balanced class weight handles minor class imbalance in default flag
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    
    # max depth restricted to 4 to avoid tree overfitting
    tree_clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    tree_clf.fit(X_train, y_train)
    
    return log_reg, tree_clf


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics, y_proba


def generate_and_save_charts(raw_df, log_reg, tree_clf, feature_names, X_test, y_test, log_metrics, tree_metrics, y_proba_log, y_proba_tree):
    sns.set_theme(style="whitegrid")
    
    # chart 1: default rate by loan purpose segment
    plt.figure(figsize=(8, 5))
    purpose_default = raw_df.groupby('loan_purpose')['default_flag'].mean().reset_index()
    purpose_default['default_rate_pct'] = purpose_default['default_flag'] * 100.0
    purpose_default = purpose_default.sort_values(by='default_rate_pct', ascending=False)
    
    ax = sns.barplot(data=purpose_default, x='loan_purpose', y='default_rate_pct', palette='crest')
    plt.title('Default Rate by Loan Purpose Segment (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Loan Purpose', fontsize=11)
    plt.ylabel('Default Rate (%)', fontsize=11)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('chart_default_rate_by_segment.png', dpi=300)
    plt.close()
    
    # chart 2: confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(log_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Non-Default', 'Default'])
    axes[0].set_yticklabels(['Non-Default', 'Default'])
    
    sns.heatmap(tree_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title('Decision Tree Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['Non-Default', 'Default'])
    axes[1].set_yticklabels(['Non-Default', 'Default'])
    plt.tight_layout()
    plt.savefig('chart_confusion_matrices.png', dpi=300)
    plt.close()

    # chart 3: roc curves
    fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {log_metrics['roc_auc']:.3f})", color='navy', lw=2)
    plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {tree_metrics['roc_auc']:.3f})", color='darkgreen', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Baseline (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('chart_roc_curves.png', dpi=300)
    plt.close()

    # chart 4: feature importances & coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    log_coef = pd.DataFrame({'feature': feature_names, 'coefficient': log_reg.coef_[0]})
    log_coef['abs_coef'] = log_coef['coefficient'].abs()
    log_coef = log_coef.sort_values(by='abs_coef', ascending=False).head(10)
    
    sns.barplot(data=log_coef, y='feature', x='coefficient', ax=axes[0], palette='vlag')
    axes[0].set_title('Top 10 Logistic Regression Coefficients', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value (Direction & Scale)')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    tree_imp = pd.DataFrame({'feature': feature_names, 'importance': tree_clf.feature_importances_})
    tree_imp = tree_imp.sort_values(by='importance', ascending=False).head(10)
    
    sns.barplot(data=tree_imp, y='feature', x='importance', ax=axes[1], palette='crest')
    axes[1].set_title('Top 10 Decision Tree Feature Importances', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig('chart_feature_importances.png', dpi=300)
    plt.close()


def run_modelling_pipeline():
    raw_df = load_modelling_data()
    df_encoded = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df_encoded)
    
    log_reg, tree_clf = train_models(X_train, y_train)
    
    log_metrics, y_proba_log = evaluate_model(log_reg, X_test, y_test, 'Logistic Regression')
    tree_metrics, y_proba_tree = evaluate_model(tree_clf, X_test, y_test, 'Decision Tree')
    
    metrics_df = pd.DataFrame([
        {
            'Model': log_metrics['model'],
            'Accuracy': f"{log_metrics['accuracy']:.4f}",
            'Precision': f"{log_metrics['precision']:.4f}",
            'Recall': f"{log_metrics['recall']:.4f}",
            'F1-Score': f"{log_metrics['f1_score']:.4f}",
            'ROC-AUC': f"{log_metrics['roc_auc']:.4f}"
        },
        {
            'Model': tree_metrics['model'],
            'Accuracy': f"{tree_metrics['accuracy']:.4f}",
            'Precision': f"{tree_metrics['precision']:.4f}",
            'Recall': f"{tree_metrics['recall']:.4f}",
            'F1-Score': f"{tree_metrics['f1_score']:.4f}",
            'ROC-AUC': f"{tree_metrics['roc_auc']:.4f}"
        }
    ])
    
    print("=" * 70)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)
    
    generate_and_save_charts(
        raw_df, log_reg, tree_clf, feature_names, X_test, y_test,
        log_metrics, tree_metrics, y_proba_log, y_proba_tree
    )
    print("saved all 4 chart visualizations successfully.")


if __name__ == "__main__":
    run_modelling_pipeline()
