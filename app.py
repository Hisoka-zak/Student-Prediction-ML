import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
import joblib
from itertools import combinations

# Directories for models and uploads
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Helper Functions
def save_file(file):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    if os.path.exists(file_path):
        print(f"Overwriting existing file: {file_path}")
    file.save(file_path)
    return file_path

def load_dataset(file_path, file_extension):
    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        return pd.read_excel(file_path, engine='openpyxl')

def identify_non_numeric_columns(df):
    """
    Identify columns in the DataFrame that cannot be converted to numeric.
    """
    non_numeric_columns = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
        except:
            non_numeric_columns.append(col)
    return non_numeric_columns

def preprocess_training_data(df):
    """
    Preprocess the training data, handling non-numeric columns, missing values,
    and outliers. Rows with non-numeric values in features are dropped.
    """
    # Drop the 'final' column if it exists
    if 'final' in df.columns:
        df = df.drop(columns=['final'])  # Drop 'final' column if it exists

    # Ensure 'grade' column exists
    if 'grade' not in df.columns:
        raise ValueError("'grade' column is required for training.")


    # Separate features (X) and target (y)
    X = df.drop(columns=['grade'])
    y = df['grade']

    # Identify non-numeric columns
    non_numeric_columns = identify_non_numeric_columns(X)
    print(f"Non-numeric columns identified: {non_numeric_columns}")
    
    for col in non_numeric_columns:
        # Replace non-numeric values with NaN, then convert
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop rows where any feature column is NaN
    X = X.dropna()

    # Align y with the updated X
    y = y.loc[X.index]

    # Drop duplicates
    X = X.drop_duplicates()
    y = y.loc[X.index]

    # Handle outliers using a relaxed IQR method
    numeric_columns = X.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        Q1 = X[col].quantile(0.05)  # Use 5th percentile
        Q3 = X[col].quantile(0.95)  # Use 95th percentile
        lower_bound = Q1
        upper_bound = Q3
        X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]

    # Ensure y aligns with the preprocessed X
    y = y.loc[X.index]

    # Check if enough samples remain
    if len(X) < 2:
        raise ValueError("Not enough valid rows remaining after preprocessing. Please review your dataset.")

    return X, y


def apply_smote(X, y):
    """
    Apply SMOTE to handle class imbalance.
    """
    class_counts = pd.Series(y).value_counts()
    smallest_class_size = class_counts.min()
    if smallest_class_size < 2:
        print("SMOTE skipped: Not enough samples in the smallest class.")
        return X, y
    smote_neighbors = min(smallest_class_size - 1, 5)
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=smote_neighbors)
    return smote.fit_resample(X, y)

def save_model_and_features(model_name, model, label_encoder, feature_names):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, f"{model_name}_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, f"{model_name}_features.pkl"))

def load_model_and_features(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"{model_name}_encoder.pkl")
    features_path = os.path.join(MODEL_DIR, f"{model_name}_features.pkl")

    if not (os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(features_path)):
        raise FileNotFoundError(f"Model, encoder, or features file not found for model: {model_name}")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)
    return model, label_encoder, feature_names

def train_models_for_subsets(course_name, X, y):
    """
    Train and save models for all possible subsets of features in the dataset.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train models for all subsets of features
    for num_features in range(1, len(X.columns) + 1):
        for feature_subset in combinations(X.columns, num_features):
            subset_X = X[list(feature_subset)]

            try:
                X_resampled, y_resampled = apply_smote(subset_X, y_encoded)
            except Exception as e:
                print(f"SMOTE skipped for subset {feature_subset}: {e}")
                X_resampled, y_resampled = subset_X, y_encoded

            # Train the model
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save the model and subset information
            subset_name = "_".join(feature_subset)
            model_name = f"{course_name}_{subset_name}"
            save_model_and_features(model_name, model, label_encoder, list(feature_subset))

    return jsonify({"message": f"Models trained successfully for {course_name}."})

# Routes
@app.route('/')
def home():
    return "Welcome to the ML Backend API! Use /upload to train or /predict to predict."

@app.route('/upload', methods=['POST'])
def upload_and_train_or_predict():
    try:
        # Retrieve file and parameters
        file = request.files.get('file')
        course_name = request.form.get('course_name', '').strip()
        prediction_only = request.form.get('prediction_only', 'false').lower() == 'true'

        # Validate inputs
        if not file or not course_name:
            return jsonify({"error": "File and course_name are required"}), 400
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['csv', 'xlsx']:
            return jsonify({"error": "Unsupported file format. Please upload a .csv or .xlsx file."}), 400

        # Save uploaded file
        file_path = save_file(file)
        df = load_dataset(file_path, file_extension)

        # Drop duplicates
        df = df.drop_duplicates()

        if prediction_only:
            # Prediction logic: 'grade' column is not required
            model_files = [
                f for f in os.listdir(MODEL_DIR) if f.startswith(f"{course_name}_") and f.endswith(".pkl")
            ]

            if not model_files:
                return jsonify({"error": f"{course_name} course is not trained yet"}), 404

            # Normalize dataset column names to lowercase
            df.columns = df.columns.str.lower()

            # Find the most suitable model based on input columns
            matched_model = None
            matched_features = None

            for model_file in model_files:
                feature_names_path = os.path.join(MODEL_DIR, model_file.replace(".pkl", "_features.pkl"))
                if os.path.exists(feature_names_path):
                    feature_names = joblib.load(feature_names_path)

                    # Normalize feature names to lowercase
                    feature_names_lower = [f.lower() for f in feature_names]

                    if set(df.columns).issubset(set(feature_names_lower)):
                        matched_model = os.path.join(MODEL_DIR, model_file)
                        matched_features = feature_names_lower
                        break

            if not matched_model:
                return jsonify({
                    "error": "No suitable model found for the uploaded dataset. Please verify your dataset columns."
                }), 400

            # Align dataset columns with the model's expected features
            df_aligned = df.reindex(columns=matched_features, fill_value=0)

            # Coerce non-numeric values to NaN and drop rows with missing values
            for col in df_aligned.columns:
                df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')

            # Drop rows containing NaN values
            df_aligned = df_aligned.dropna()

            # Check if the dataset is empty after dropping rows
            if df_aligned.empty:
                return jsonify({"error": "All rows were dropped due to invalid or missing values. Please review your dataset."}), 400

            # Load the matched model and label encoder
            model = joblib.load(matched_model)
            label_encoder = joblib.load(matched_model.replace(".pkl", "_encoder.pkl"))

            # Perform predictions
            predictions = model.predict(df_aligned)
            predicted_grades = label_encoder.inverse_transform([int(round(pred)) for pred in predictions])
            df_aligned['Prediction'] = predicted_grades

            # Return the dataset with predictions
            return jsonify({"predictions": df_aligned.to_dict(orient='records')})

        else:
            # Training logic: 'grade' column is required
            if 'grade' not in df.columns.str.lower():
                return jsonify({"error": "'grade' column is required in the dataset for training."}), 400

            # Normalize column names before training
            df.columns = df.columns.str.lower()
            X, y = preprocess_training_data(df)
            return train_models_for_subsets(course_name, X, y)


    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        course_name = data.get('course_name', '').strip()
        inputs = data.get('inputs')

        if not course_name or not inputs:
            return jsonify({"error": "course_name and inputs are required"}), 400

        try:
            # List all models available for this course
            model_files = [
                f for f in os.listdir(MODEL_DIR) if f.startswith(f"{course_name}_") and f.endswith(".pkl")
            ]

            if not model_files:
                return jsonify({"error": f"No models found for course: {course_name}"}), 404

            # Match input keys with available models
            matched_model = None
            matched_features = None

            for model_file in model_files:
                feature_names_path = os.path.join(MODEL_DIR, model_file.replace(".pkl", "_features.pkl"))
                if os.path.exists(feature_names_path):
                    feature_names = joblib.load(feature_names_path)
                    if set(inputs.keys()).issubset(set(feature_names)):
                        matched_model = os.path.join(MODEL_DIR, model_file)
                        matched_features = feature_names
                        break

            if not matched_model:
                return jsonify({
                    "error": "No suitable model found for the provided assessments. Please ensure your input matches available assessments."
                }), 400

            # Load the matched model and label encoder
            model = joblib.load(matched_model)
            label_encoder = joblib.load(matched_model.replace(".pkl", "_encoder.pkl"))

            # Prepare input data
            inputs_df = pd.DataFrame([inputs]).reindex(columns=matched_features, fill_value=0)
            inputs_df = inputs_df.fillna(inputs_df.mean())

            # Perform prediction
            predictions = model.predict(inputs_df)
            predictions = label_encoder.inverse_transform([int(round(pred)) for pred in predictions])

            return jsonify({"predictions": predictions.tolist()})

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": f"Please fill in fields to predict or hide unnecessary assessment"}), 500

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
