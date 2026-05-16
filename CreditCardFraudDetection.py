import pandas as pd
from imblearn.over_sampling import SMOTE #oversampling = more processing but better accuracy!!
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # A little faster than random forest (Spolier alert, this didn't work for our real life example)  

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier #no more cutting corners let's put the CPU power into use and use the beast


# LoaDING the Training dataset
df = pd.read_csv('fraudTrain.csv')

#  readable columns
print("--- First 5 Rows ---")
print(df[['merchant', 'category', 'amt', 'is_fraud']].head(), "\n")

# exact imbalance
print("--- Class Distribution ---")
class_counts = df['is_fraud'].value_counts()
class_percentages = df['is_fraud'].value_counts(normalize=True) * 100

print(f"Normal Transactions (0): {class_counts[0]} ({class_percentages[0]:.3f}%)")
print(f"Fraudulent Transactions (1): {class_counts[1]} ({class_percentages[1]:.3f}%)")

# PREPROCESSING (Getting Rid of info that we don't need to keep our math clean)

#  DropPING text columns
# 'merchant' has thousands of unique names, which will crash your memory if we encode it.

df_clean = df.drop('merchant', axis=1, errors='ignore')

#  Convert 'category' into numbers (One-Hot Encoding)
# This creates a new column for every category with a 1 or 0 (e.g., is_grocery_pos: 1)
df_encoded = pd.get_dummies(df_clean, columns=['category'])

#  Drop any other remaining text columns (like names or addresses)
#  ready for math.
df_encoded = df_encoded.select_dtypes(exclude=['object'])

# Separatingg Features (X) from Target (y)
# X = The clues (Amount, Category, etc.)
# y = The answer key (Is it fraud or not?)
X = df_encoded.drop('is_fraud', axis=1)
y = df_encoded['is_fraud']

print("\n--- Preprocessing Complete ---")
print(f"Features (X) shape: {X.shape}")
print("All data is now purely numerical!")




print("--- Before Oversampling ---")
print(y.value_counts())

# the SMOTE algorithm
# random_state=42 ensures you get the exact same synthetic data every time you run it
smote = SMOTE(random_state=42)

# balance the scales

print("\nGenerating synthetic fraud transactions... Please wait.")
X_balanced, y_balanced = smote.fit_resample(X, y)

# Prove the balance
print("\n--- After Oversampling ---")
print(y_balanced.value_counts())
print(f"\nNew Features (X) shape: {X_balanced.shape}")


# Let's start testing with SMOTE using Logistic Regression

print("\n--- Splitting Data ---")
# Splitting the data
# We keep 20% hidden from the AI during training so we can give it a pop quiz later.
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
print("Data split complete.")

# Initialize the AI model
# max_iter=1000 gives the math enough time to settle on the right answer
model = LogisticRegression(max_iter=1000)

# Train the model (This is the actual "Machine Learning" part)
print("\nTraining Logistic Regression... Please wait, this may take 1-2 minutes.")
model.fit(X_train, y_train)

# Give the AI its pop quiz using the 20% of data we hid earlier
print("\nTesting the AI...")
y_pred = model.predict(X_val)

# Print the results!
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_val, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred))

print("\n--- Fixing the Data: Feature Scaling ---")
scaler = StandardScaler()

# MY Model wasn't doing very well and didn't catch any theieves on the test so we will scale the data using scaler to force it to do better. 

# then we transform both the training and validation sets.

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print("Features scaled successfully.")

# Re-initialize the AI model
model_scaled = LogisticRegression(max_iter=1000)

# Training the model on the mathematically balanced data
print("\nTraining Logistic Regression on SCALED data... Please wait.")
model_scaled.fit(X_train_scaled, y_train)

# Give the AI the pop quiz again
print("\nTesting the AI...")
y_pred_scaled = model_scaled.predict(X_val_scaled)

# printing the new results!
print("\n--- NEW Confusion Matrix ---")
print(confusion_matrix(y_val, y_pred_scaled))

print("\n--- NEW Classification Report ---")
print(classification_report(y_val, y_pred_scaled))





print("           THE FINAL EXAM  USING Logistic Regression (Turned out to be not the final final exam:)    ")


# Load the completely unseen Test dataset
df_test = pd.read_csv('fraudTest.csv')

# Preprocess exactly the same way as the training data
df_test_clean = df_test.drop('merchant', axis=1, errors='ignore')
df_test_encoded = pd.get_dummies(df_test_clean, columns=['category'])
df_test_encoded = df_test_encoded.select_dtypes(exclude=['object'])

# CRITICAL Aligning Columns
# In the real world, a test dataset might be missing a specific 'category' that was 
# present in the training data. This line forces the test data to have the exact same columns as original training data, filling any missing ones with 0.

df_test_encoded = df_test_encoded.reindex(columns=df_encoded.columns, fill_value=0)

# Separating Features (X) and Target (y)

X_test = df_test_encoded.drop('is_fraud', axis=1)
y_test = df_test_encoded['is_fraud']
print(f"Test data loaded and preprocessed. Shape: {X_test.shape}")

# Scaling the test data
# We use scaler.transform(), NOT fit_transform(). We want to use the exact same mathematical ruler we built during training so the AI isn't confused.

print("Scaling test features...")
X_test_scaled = scaler.transform(X_test)

# Take the Test
print("Running the AI model on the unseen data...")
y_test_pred = model_scaled.predict(X_test_scaled)

# The Real-World Results
print("\n--- FINAL Confusion Matrix ---")
print(confusion_matrix(y_test, y_test_pred))

print("\n--- FINAL Classification Report ---")
print(classification_report(y_test, y_test_pred))



print("\n========================================")
print("      THE UPGRADE: RANDOM FOREST        ")
print("========================================\n")

# Initialize the Random Forest
# n_estimators=100: 100 distinct decision trees
# n_jobs=-1: Forces the computer to use ALL available CPU cores for maximum speed
# random_state=42: Ensures reproducibility


print("Initializing Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# Training the Model
print("Training the Random Forest... (Grab a coffee or wash your dishes, this will take a few minutes!)")
rf_model.fit(X_train_scaled, y_train)

# Final Exam directly on the unseen Test Data
print("\nTesting the Random Forest on the real-world data...")
y_test_pred_rf = rf_model.predict(X_test_scaled)

# The Real-World Results
print("\n--- RANDOM FOREST: FINAL Confusion Matrix ---")
print(confusion_matrix(y_test, y_test_pred_rf))

print("\n--- RANDOM FOREST: FINAL Classification Report ---")
print(classification_report(y_test, y_test_pred_rf))
