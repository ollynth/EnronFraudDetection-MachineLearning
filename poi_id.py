#!/usr/bin/python

import sys
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(("./tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

### Task 1: Select what features you'll use.
### Define initial feature lists - these will be refined through feature selection
# Financial features available in the dataset
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                     'bonus', 'restricted_stock_deferred', 'deferred_income', 
                     'total_stock_value', 'expenses', 'exercised_stock_options', 
                     'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

# Email communication features available in the dataset
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Combine all features with 'poi' as the target variable
features_list = ['poi'] + financial_features + email_features

### Load the dataset from pickle file
with open("./final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Display basic dataset information
print("Dataset Overview:")
print(f"Total number of people: {len(data_dict)}")
print(f"Total number of POIs: {sum(1 for person in data_dict.values() if person['poi'])}")
print(f"Total features available: {len(features_list)}")

### Task 2: Remove outliers
print("\nOutlier Investigation:")

# Remove known outliers that are not actual people
outliers_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers_to_remove:
    if outlier in data_dict:
        print(f"Removing outlier: {outlier}")
        data_dict.pop(outlier)  # Remove outlier from dataset

# Analyze missing data patterns across features
missing_data_count = {}
for feature in features_list[1:]:  # Skip 'poi' target variable
    missing_count = sum(1 for person in data_dict.values() if person[feature] == 'NaN')
    missing_data_count[feature] = missing_count

print(f"Total people after outlier removal: {len(data_dict)}")

### Task 3: Create new feature(s)
print("\nFeature Engineering:")

def create_new_features(data_dict):
    """
    Create new features from existing ones to potentially improve classification
    These features capture relationships between existing features
    """
    for person in data_dict:
        # Feature 1: Calculate fraction of emails received from POIs
        # This measures how much communication comes from POIs
        if (data_dict[person]['from_poi_to_this_person'] != 'NaN' and 
            data_dict[person]['to_messages'] != 'NaN' and 
            data_dict[person]['to_messages'] > 0):
            data_dict[person]['fraction_from_poi'] = (
                data_dict[person]['from_poi_to_this_person'] / 
                data_dict[person]['to_messages']
            )
        else:
            data_dict[person]['fraction_from_poi'] = 0
            
        # Feature 2: Calculate fraction of emails sent to POIs
        # This measures how much someone communicates with POIs
        if (data_dict[person]['from_this_person_to_poi'] != 'NaN' and 
            data_dict[person]['from_messages'] != 'NaN' and 
            data_dict[person]['from_messages'] > 0):
            data_dict[person]['fraction_to_poi'] = (
                data_dict[person]['from_this_person_to_poi'] / 
                data_dict[person]['from_messages']
            )
        else:
            data_dict[person]['fraction_to_poi'] = 0
            
        # Feature 3: Calculate ratio of stock value to total payments
        # This measures the proportion of compensation that comes from stock
        if (data_dict[person]['total_payments'] != 'NaN' and 
            data_dict[person]['total_stock_value'] != 'NaN' and
            data_dict[person]['total_payments'] > 0):
            data_dict[person]['stock_to_payment_ratio'] = (
                data_dict[person]['total_stock_value'] / 
                data_dict[person]['total_payments']
            )
        else:
            data_dict[person]['stock_to_payment_ratio'] = 0

# Apply feature engineering to the dataset
create_new_features(data_dict)

# Add new features to the feature list
new_features = ['fraction_from_poi', 'fraction_to_poi', 'stock_to_payment_ratio']
features_list_extended = features_list + new_features

print(f"Created {len(new_features)} new features: {new_features}")

### Store dataset for easy export
my_dataset = data_dict

### Task 4: Feature Selection
print("\nFeature Selection:")

# Convert dataset to format suitable for sklearn
data = featureFormat(my_dataset, features_list_extended, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Use SelectKBest to identify the most informative features
# f_classif computes ANOVA F-statistic for feature selection
selector = SelectKBest(f_classif, k=5)  # Select top 10 features
features_selected = selector.fit_transform(features, labels)
feature_scores = selector.scores_
feature_names = features_list_extended[1:]  # Remove 'poi' from feature names

# Get the indices and names of selected features
selected_indices = selector.get_support(indices=True)
selected_features = ['poi'] + [feature_names[i] for i in selected_indices]
selected_scores = [feature_scores[i] for i in selected_indices]

# Display selected features and their scores
print("Top 5 features selected:")
for feature, score in zip(selected_features[1:], selected_scores):
    print(f"{feature}: {score:.2f}")

# Update features_list with selected features
features_list = selected_features

### Extract final features and labels for model training
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Test Decision Tree Classifier
print("\nTesting Decision Tree Classifier:")

# Split data into training and testing sets for initial evaluation
# stratify=labels ensures balanced representation of POIs and non-POIs
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Use StratifiedShuffleSplit for robust cross-validation
# This ensures balanced class distribution across folds
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.1, random_state=42)

# # === MODEL 1: GaussianNB with GridSearchCV ===
from sklearn.naive_bayes import GaussianNB
gnb_init = GaussianNB()
gnb_init.fit(features_train, labels_train)
# Predict on test set
predictions = gnb_init.predict(features_test)
# Calculate performance metrics for initial classifier
precision = precision_score(labels_test, predictions, zero_division=0)
recall = recall_score(labels_test, predictions, zero_division=0)
f1 = f1_score(labels_test, predictions, zero_division=0)
print(f"Initial GaussianNB - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print(f"\nTuning GausianNB Classifier...")
param_grid_gnb = {
    'var_smoothing': np.logspace(0, -9, num=10)
}
gnb = GaussianNB()
grid_gnb = GridSearchCV(gnb, param_grid_gnb, cv=sss, scoring='f1')
grid_gnb.fit(features_train, labels_train)
best_gnb = grid_gnb.best_estimator_
gnb_best_predictions = best_gnb.predict(features_test)
print("\n=== GaussianNB Classification Report ===")
print(classification_report(labels_test, gnb_best_predictions))
print("Best GaussianNB params:", grid_gnb.best_params_)
# Calculate performance metrics for best classifier
precision = precision_score(labels_test, gnb_best_predictions, zero_division=0)
recall = recall_score(labels_test, gnb_best_predictions, zero_division=0)   
f1 = f1_score(labels_test, gnb_best_predictions, zero_division=0)

print(f"Tuning Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
#  display confusion matrix
cm = confusion_matrix(labels_test, gnb_best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Non-POI', 'POI'], yticklabels=['Non-POI', 'POI'])
plt.title('Confusion Matrix for GaussianNB')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("===========================================")

# === MODEL 2: RandomForestClassifier with GridSearchCV ===
from sklearn.ensemble import RandomForestClassifier
rf_init = RandomForestClassifier(random_state=42, n_estimators=100)
rf_init.fit(features_train, labels_train)
# Predict on test set
predictions = rf_init.predict(features_test)
# Calculate performance metrics for best classifier
precision = precision_score(labels_test, predictions, zero_division=0)
recall = recall_score(labels_test, predictions, zero_division=0)
f1 = f1_score(labels_test, predictions, zero_division=0)
print(f"Initial RandomForest - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print(f"\nTuning Random Forest Classifier...")
param_grid_rf = {
    'n_estimators': [5, 10],
    'max_depth': [None, 2, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf,  cv=sss, scoring='f1')
grid_rf.fit(features_train, labels_train)
best_rf = grid_rf.best_estimator_
rf_best_predictions = best_rf.predict(features_test)
print("\n=== Random Forest Classification Report ===")
print(classification_report(labels_test, rf_best_predictions))
print("Best RandomForest params:", grid_rf.best_params_)
# Calculate performance metrics for best classifier
precision = precision_score(labels_test, predictions, zero_division=0)
recall = recall_score(labels_test, predictions, zero_division=0)
f1 = f1_score(labels_test, predictions, zero_division=0)
print(f"Tuning Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
# Display confusion matrix
cm = confusion_matrix(labels_test, rf_best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=['Non-POI', 'POI'], yticklabels=['Non-POI', 'POI'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("===========================================")

# # === MODEL 3: DecisionTree  ===
dt_init = DecisionTreeClassifier(random_state=42)
dt_init.fit(features_train, labels_train)
predictions = dt_init.predict(features_test)
# Calculate performance metrics for initial classifier
precision = precision_score(labels_test, predictions, zero_division=0)
recall = recall_score(labels_test, predictions, zero_division=0)
f1 = f1_score(labels_test, predictions, zero_division=0)
print(f"Initial Decision Tree - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

### Task 5: Tune Decision Tree Classifier
print(f"\nTuning Decision Tree Classifier...")
param_grid_dt = {
    'min_samples_split': [2, 5, 10, 15, 20],      # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 5, 10],            # Minimum samples required at a leaf node
    'max_depth': [None, 3, 5, 10, 15],            # Maximum depth of the tree
    'criterion': ['gini', 'entropy'],              # Splitting criterion
    'max_features': [None, 'sqrt', 'log2']        # Number of features to consider for splits
}

# Perform grid search to find best hyperparameters
# f1 score is used as it balances precision and recall
grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42), 
    param_grid_dt, 
    cv=sss, 
    scoring='f1'
)
grid_dt.fit(features_train, labels_train)
best_dt = grid_dt.best_estimator_ 
dt_best_predictions = best_dt.predict(features_test)
print("\n=== Decision Tree Classification Report ===")
print(classification_report(labels_test, dt_best_predictions))
# Calculate performance metrics for best classifier
precision = precision_score(labels_test, dt_best_predictions, zero_division=0)
recall = recall_score(labels_test, dt_best_predictions, zero_division=0)
f1 = f1_score(labels_test, dt_best_predictions, zero_division=0)
print(f"Tuning Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
# Display confusion matrix
cm = confusion_matrix(labels_test, dt_best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=['Non-POI', 'POI'], yticklabels=['Non-POI', 'POI'])
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Display best parameters and cross-validation score
print(f"Best parameters: {grid_dt.best_params_}")
print(f"Best cross-validation score: {grid_dt.best_score_:.3f}")
print("===========================================")

### Task 6: Final Evaluation
print("\nFinal Evaluation using tester.py:")

# choose best classifier based on F1 score
clf_list =[best_dt, best_gnb, best_rf]
# # Select the best classifier based on F1 score
clf = max(clf_list, key=lambda x: f1_score(labels_test, x.predict(features_test), zero_division=0))
print(f"Selected best classifier: {clf.__class__.__name__}")

# Try to use the provided tester.py for evaluation
try:
    test_classifier(clf, my_dataset, features_list)
except TypeError as e:
    # Handle compatibility issues with tester.py
    print(f"tester.py compatibility issue: {e}")
    print("Running manual evaluation instead...")
    
    # Perform manual evaluation using multiple train-test splits
    sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state=42)
    
    # Lists to store scores from each iteration
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Evaluate classifier across multiple random splits
    for train_idx, test_idx in sss.split(features, labels):
        features_train = [features[i] for i in train_idx]
        features_test = [features[i] for i in test_idx]
        labels_train = [labels[i] for i in train_idx]
        labels_test = [labels[i] for i in test_idx]
        
        # Train and predict on this split
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        
        # Calculate and store metrics
        precision_scores.append(precision_score(labels_test, predictions, zero_division=0))
        recall_scores.append(recall_score(labels_test, predictions, zero_division=0))
        f1_scores.append(f1_score(labels_test, predictions, zero_division=0))
    
    # Calculate average performance across all splits
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    
    # Display final evaluation results
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1-Score: {avg_f1:.3f}")
    
    # Check if classifier meets project requirements
    if avg_precision >= 0.3 and avg_recall >= 0.3:
        print("✓ Classifier meets project requirements (Precision ≥ 0.3, Recall ≥ 0.3)")
    else:
        print("✗ Classifier does not meet project requirements")

# Analyze feature importance from the trained Decision Tree
print("\nFeature Importances:")
feature_importance = clf.feature_importances_

# Create DataFrame for better visualization of feature importance
importance_df = pd.DataFrame({
    'feature': features_list[1:],  # Exclude 'poi' target variable
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df)

# Create visualizations for feature importance
plt.figure(figsize=(12, 8))

# Filter features with meaningful importance (> 0.01)
important_features = importance_df[importance_df['importance'] > 0.01]

# Horizontal bar chart of feature importance
plt.subplot(2, 1, 1)
plt.barh(important_features['feature'], important_features['importance'])
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  # Show highest importance at top

# Pie chart for top 6 features
plt.subplot(2, 1, 2)
top_features = important_features.head(6)
plt.pie(top_features['importance'], labels=top_features['feature'], autopct='%1.1f%%')
plt.title('Top 6 Feature Importance Distribution')

plt.tight_layout()
plt.savefig('decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary results table
summary_data = {
    'Metric': ['Total People', 'Total POIs', 'Features Used', 'Final Precision', 'Final Recall', 'Final F1-Score'],
    'Value': [len(data_dict), sum(1 for person in data_dict.values() if person['poi']), 
              len(features_list), f"{avg_precision:.3f}", f"{avg_recall:.3f}", f"{avg_f1:.3f}"]
}

summary_df = pd.DataFrame(summary_data)
print("\nProject Summary:")
print(summary_df.to_string(index=False))

# Create comprehensive results visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pie chart showing dataset distribution (POIs vs non-POIs)
dataset_info = pd.DataFrame({
    'Category': ['Total People', 'POIs', 'Non-POIs'],
    'Count': [len(data_dict), 
              sum(1 for person in data_dict.values() if person['poi']),
              len(data_dict) - sum(1 for person in data_dict.values() if person['poi'])]
})

axes[0].pie(dataset_info['Count'][1:], labels=['POIs', 'Non-POIs'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Dataset Distribution')

# Bar chart showing final performance metrics
final_metrics = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'],
    'Score': [avg_precision, avg_recall, avg_f1]
})

bars = axes[1].bar(final_metrics['Metric'], final_metrics['Score'], color=['#ff9999', '#66b3ff', '#99ff99'])
axes[1].set_title('Decision Tree Performance')
axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1)

# Add score labels on top of bars
for bar, score in zip(bars, final_metrics['Score']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')

# Bar chart comparing achieved scores vs project requirements
requirements = pd.DataFrame({
    'Requirement': ['Precision ≥ 0.3', 'Recall ≥ 0.3'],
    'Achieved': [avg_precision, avg_recall],
    'Target': [0.3, 0.3]
})

x = np.arange(len(requirements))
width = 0.35

axes[2].bar(x - width/2, requirements['Achieved'], width, label='Achieved', alpha=0.8)
axes[2].bar(x + width/2, requirements['Target'], width, label='Target', alpha=0.8)
axes[2].set_xlabel('Requirements')
axes[2].set_ylabel('Score')
axes[2].set_title('Project Requirements Check')
axes[2].set_xticks(x)
axes[2].set_xticklabels(requirements['Requirement'])
axes[2].legend()
axes[2].set_ylim(0, max(1, max(requirements['Achieved']) + 0.1))

plt.tight_layout()
plt.savefig('decision_tree_final_results.png', dpi=300, bbox_inches='tight')
plt.show()

### Export classifier, dataset, and features for submission
print(f"\nExporting Decision Tree classifier with {len(features_list)} features...")

# Try to use the provided dump function
try:
    dump_classifier_and_data(clf, my_dataset, features_list)
except TypeError as e:
    # Handle pickle compatibility issues with manual export
    print(f"tester.py pickle compatibility issue: {e}")
    print("Using manual export instead...")
    
    # Manual export using binary mode (required for pickle)
    with open("my_classifier.pkl", "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    
    with open("my_dataset.pkl", "wb") as dataset_outfile:
        pickle.dump(my_dataset, dataset_outfile)
    
    with open("my_feature_list.pkl", "wb") as featurelist_outfile:
        pickle.dump(features_list, featurelist_outfile)
    
    print("Decision Tree classifier, dataset, and features exported manually.")

# Final project completion summary
print("\nProject completed successfully!")
print(f"Final features used: {features_list}")
print("Generated visualizations: decision_tree_feature_importance.png, decision_tree_final_results.png")
