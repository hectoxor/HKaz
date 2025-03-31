# Hong Kong Fitness Program Adherence Prediction Model
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
import folium
from folium.plugins import HeatMap
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection and Preprocessing
def load_fitness_survey_data(filepath):
    """
    Load and preprocess LCSD fitness survey data
    """
    # Simulating data loading as we don't have actual data files
    print(f"Loading data from {filepath}")
    
    # Create synthetic dataset based on survey information in 1fit.md
    np.random.seed(42)
    n_samples = 9300  # Sample size mentioned in the document
    
    # Demographic features
    age_groups = ['7-12', '13-19', '20-39', '40-59', '60-79']
    age_weights = [0.15, 0.15, 0.3, 0.25, 0.15]  # Distribution weights
    ages = np.random.choice(age_groups, size=n_samples, p=age_weights)
    
    gender = np.random.choice(['M', 'F'], size=n_samples)
    
    # 18 districts in Hong Kong
    districts = [
        'Central & Western', 'Wan Chai', 'Eastern', 'Southern', 
        'Yau Tsim Mong', 'Sham Shui Po', 'Kowloon City', 'Wong Tai Sin', 
        'Kwun Tong', 'Kwai Tsing', 'Tsuen Wan', 'Tuen Mun', 
        'Yuen Long', 'North', 'Tai Po', 'Sha Tin', 
        'Sai Kung', 'Islands'
    ]
    # Some districts have higher population density
    district_weights = [
        0.05, 0.05, 0.07, 0.05, 0.08, 0.07, 0.06, 0.06, 
        0.08, 0.05, 0.05, 0.06, 0.06, 0.04, 0.04, 0.07, 
        0.04, 0.02
    ]
    districts_sampled = np.random.choice(districts, size=n_samples, p=district_weights)
    
    # Income levels (for considering bias mitigation)
    income_levels = ['Low', 'Medium', 'High']
    # Higher income in certain districts
    income_prob = []
    for district in districts_sampled:
        if district in ['Central & Western', 'Wan Chai', 'Southern']:
            income_prob.append([0.2, 0.3, 0.5])  # Higher probability of high income
        elif district in ['Sham Shui Po', 'Wong Tai Sin', 'Kwun Tong']:
            income_prob.append([0.5, 0.3, 0.2])  # Higher probability of low income
        else:
            income_prob.append([0.3, 0.4, 0.3])  # Balanced
    
    income = [np.random.choice(income_levels, p=prob) for prob in income_prob]
    
    # Behavioral features
    exercise_frequency = []
    for age_group in ages:
        if age_group in ['7-12', '13-19']:
            # Children have varied patterns due to school
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.2, 0.3, 0.2, 0.2, 0.1]
            )
        elif age_group in ['20-39']:
            # Young adults cite "lack of time"
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.3, 0.3, 0.2, 0.15, 0.05]
            )
        elif age_group in ['40-59']:
            # Middle-aged adults
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.25, 0.35, 0.2, 0.15, 0.05]
            )
        else:
            # Older adults might have more time but less mobility
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.3, 0.2, 0.2, 0.2, 0.1]
            )
        exercise_frequency.append(freq)
    
    # Exercise types
    exercise_types = []
    for gender_val, age_group in zip(gender, ages):
        type_probs = []
        if age_group in ['7-12', '13-19']:
            # Children more likely to do sports
            type_probs = [0.4, 0.2, 0.2, 0.1, 0.1]
        elif gender_val == 'M':
            # Males more likely to do strength training
            type_probs = [0.2, 0.3, 0.15, 0.25, 0.1]
        else:
            # Females more likely to do aerobic/yoga
            type_probs = [0.2, 0.15, 0.3, 0.15, 0.2]
        
        exercise_types.append(np.random.choice(
            ['Sports', 'Strength', 'Aerobic', 'Walking/Jogging', 'Yoga/Flexibility'],
            p=type_probs
        ))
    
    # Physiological features
    muscular_strength = []
    for age_group, gender_val, freq in zip(ages, gender, exercise_frequency):
        base_score = {
            '7-12': 60,
            '13-19': 65,
            '20-39': 70,
            '40-59': 65,
            '60-79': 50
        }[age_group]
        
        # Gender adjustment (slight difference in baseline strength)
        if gender_val == 'M':
            base_score += 5
        
        # Exercise frequency adjustment
        freq_adjustment = {
            'Never': -15,
            'Rarely': -5,
            'Sometimes': 0,
            'Regularly': 10,
            'Daily': 15
        }[freq]
        
        # Add some noise
        final_score = base_score + freq_adjustment + np.random.normal(0, 8)
        
        # Clamp between 0 and 100
        muscular_strength.append(max(0, min(100, final_score)))
    
    # BMI values (simulate obesity markers)
    bmi_values = []
    for age_group, freq in zip(ages, exercise_frequency):
        if freq in ['Never', 'Rarely']:
            base_bmi = 26 + np.random.normal(0, 3)  # Higher baseline for sedentary
        else:
            base_bmi = 22 + np.random.normal(0, 2)  # Lower baseline for active
            
        # Age adjustments
        if age_group in ['7-12']:
            base_bmi -= 4  # Children typically have lower BMI
        elif age_group in ['60-79']:
            base_bmi += 1  # Older adults often have slightly higher BMI
            
        bmi_values.append(max(15, min(40, base_bmi)))  # Clamp to realistic range
    
    # Create synthetic coordinates for each district for GIS integration
    # These are approximate center points for Hong Kong districts
    district_coords = {
        'Central & Western': (22.2826, 114.1452),
        'Wan Chai': (22.2808, 114.1826),
        'Eastern': (22.2845, 114.2256),
        'Southern': (22.2458, 114.1600),
        'Yau Tsim Mong': (22.3203, 114.1694),
        'Sham Shui Po': (22.3303, 114.1622),
        'Kowloon City': (22.3287, 114.1839),
        'Wong Tai Sin': (22.3419, 114.1953),
        'Kwun Tong': (22.3100, 114.2260),
        'Kwai Tsing': (22.3561, 114.1324),
        'Tsuen Wan': (22.3725, 114.1170),
        'Tuen Mun': (22.3908, 113.9725),
        'Yuen Long': (22.4445, 114.0225),
        'North': (22.4940, 114.1386),
        'Tai Po': (22.4513, 114.1644),
        'Sha Tin': (22.3864, 114.1928),
        'Sai Kung': (22.3809, 114.2707),
        'Islands': (22.2627, 113.9456)
    }
    
    # Add some noise to coordinates to simulate addresses in each district
    lats = []
    longs = []
    for district in districts_sampled:
        base_lat, base_long = district_coords[district]
        # Add noise (approximately within a 1-2km radius)
        lat = base_lat + np.random.normal(0, 0.01)
        long = base_long + np.random.normal(0, 0.01)
        lats.append(lat)
        longs.append(long)
    
    # Synthesize fitness track data
    fitness_tracks = [
        {'name': 'Victoria Park Jogging Trail', 'coords': [(22.2824, 114.1884), (22.2836, 114.1892)]},
        {'name': 'Kowloon Park Fitness Trail', 'coords': [(22.3023, 114.1693), (22.3045, 114.1698)]},
        {'name': 'Sha Tin Shing Mun River Track', 'coords': [(22.3794, 114.1878), (22.3825, 114.1901)]},
        {'name': 'Tseung Kwan O Waterfront Track', 'coords': [(22.3067, 114.2608), (22.3102, 114.2638)]},
        {'name': 'Tai Po Waterfront Park Track', 'coords': [(22.4502, 114.1711), (22.4530, 114.1732)]},
    ]
    
    # Calculate distance to nearest fitness track
    distances_to_track = []
    for lat, long in zip(lats, longs):
        min_distance = float('inf')
        for track in fitness_tracks:
            for track_coord in track['coords']:
                dist = haversine((lat, long), track_coord)
                min_distance = min(min_distance, dist)
        distances_to_track.append(min_distance)
    
    # Target variable: program adherence 
    # Factors that increase adherence probability:
    # - Regular exercise frequency
    # - Younger adults and middle-aged (20-59)
    # - Proximity to fitness tracks
    # - Better muscular strength scores
    
    adherence_prob = []
    for freq, age, dist, strength, bmi in zip(
        exercise_frequency, ages, distances_to_track, muscular_strength, bmi_values
    ):
        # Base probability
        prob = 0.5
        
        # Exercise frequency impact
        freq_factor = {
            'Never': -0.25,
            'Rarely': -0.15,
            'Sometimes': 0,
            'Regularly': 0.15,
            'Daily': 0.25
        }[freq]
        prob += freq_factor
        
        # Age impact
        if age in ['20-39', '40-59']:
            prob += 0.05
        elif age in ['7-12']:
            prob += 0.1  # Children often follow programs in school
        elif age in ['60-79']:
            prob -= 0.05  # Older adults might face mobility issues
            
        # Distance impact - closer is better
        if dist < 2:  # Within 2km
            prob += 0.1
        elif dist > 5:  # More than 5km away
            prob -= 0.1
            
        # Fitness level impact
        if strength > 70:
            prob += 0.05  # Already fit people might stick with programs
        elif strength < 40:
            prob -= 0.05  # Very unfit people might give up easier
            
        # BMI impact
        if bmi > 30:  # Obese
            prob -= 0.05  # Might face more challenges
            
        # Add some randomness
        prob += np.random.normal(0, 0.1)
        
        # Clamp between 0 and 1
        prob = max(0, min(1, prob))
        adherence_prob.append(prob)
    
    # Binary outcome based on probability
    adherence = [1 if p > 0.5 else 0 for p in adherence_prob]
    
    # Create the dataframe
    data = pd.DataFrame({
        'age_group': ages,
        'gender': gender,
        'district': districts_sampled,
        'income_level': income,
        'exercise_frequency': exercise_frequency,
        'exercise_type': exercise_types,
        'muscular_strength': muscular_strength,
        'bmi': bmi_values,
        'latitude': lats,
        'longitude': longs,
        'distance_to_track': distances_to_track,
        'adherence_probability': adherence_prob,
        'adherence': adherence
    })
    
    print(f"Created synthetic dataset with {len(data)} samples")
    print(f"Adherence rate: {data['adherence'].mean():.2f}")
    
    return data

def load_gis_data():
    """
    Load GIS data for fitness tracks
    In a real implementation, this would load from GIS files
    """
    # For simplicity, we'll create synthetic GIS data
    # In a real implementation, you'd use geopandas to load shapefiles
    
    fitness_tracks = [
        {'name': 'Victoria Park Jogging Trail', 
         'length_km': 1.2, 
         'elevation_gain': 5,
         'coords': [(22.2824, 114.1884), (22.2836, 114.1892)]},
        {'name': 'Kowloon Park Fitness Trail', 
         'length_km': 0.8, 
         'elevation_gain': 15,
         'coords': [(22.3023, 114.1693), (22.3045, 114.1698)]},
        {'name': 'Sha Tin Shing Mun River Track', 
         'length_km': 2.5, 
         'elevation_gain': 0,
         'coords': [(22.3794, 114.1878), (22.3825, 114.1901)]},
        {'name': 'Tseung Kwan O Waterfront Track', 
         'length_km': 3.0, 
         'elevation_gain': 5,
         'coords': [(22.3067, 114.2608), (22.3102, 114.2638)]},
        {'name': 'Tai Po Waterfront Park Track', 
         'length_km': 1.5, 
         'elevation_gain': 10,
         'coords': [(22.4502, 114.1711), (22.4530, 114.1732)]},
    ]
    
    return pd.DataFrame(fitness_tracks)

# 2. Feature Engineering
def engineer_features(df, gis_data):
    """
    Create derived features from raw data
    """
    # Create age as numeric feature
    age_mapping = {
        '7-12': 10,
        '13-19': 16,
        '20-39': 30,
        '40-59': 50,
        '60-79': 70
    }
    df['age_numeric'] = df['age_group'].map(age_mapping)
    
    # Create exercise frequency as numeric
    freq_mapping = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Regularly': 3,
        'Daily': 4
    }
    df['exercise_frequency_numeric'] = df['exercise_frequency'].map(freq_mapping)
    
    # Create income level as numeric
    income_mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    df['income_numeric'] = df['income_level'].map(income_mapping)
    
    # Create BMI category
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    
    # Distance to track buckets
    def distance_category(dist):
        if dist < 1:
            return 'Very Close (<1km)'
        elif dist < 3:
            return 'Close (1-3km)'
        elif dist < 5:
            return 'Moderate (3-5km)'
        else:
            return 'Far (>5km)'
    
    df['distance_category'] = df['distance_to_track'].apply(distance_category)
    
    # Muscular strength categories
    def strength_category(score):
        if score < 40:
            return 'Low'
        elif score < 70:
            return 'Medium'
        else:
            return 'High'
    
    df['strength_category'] = df['muscular_strength'].apply(strength_category)
    
    # Create interaction features
    df['age_exercise_interaction'] = df['age_numeric'] * df['exercise_frequency_numeric']
    df['age_bmi_interaction'] = df['age_numeric'] * df['bmi']
    
    return df

# 3. Model Building
def build_model(X_train, y_train, use_smote=True, optimize_hyperparams=True):
    """
    Build and train the random forest model
    """
    # Define categorical and numerical features
    categorical_features = [
        'age_group', 'gender', 'district', 'income_level', 
        'exercise_frequency', 'exercise_type', 'bmi_category',
        'distance_category', 'strength_category'
    ]
    
    numerical_features = [
        'muscular_strength', 'bmi', 'distance_to_track',
        'age_numeric', 'exercise_frequency_numeric', 'income_numeric',
        'age_exercise_interaction', 'age_bmi_interaction'
    ]
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Handle class imbalance with SMOTE if specified
    if use_smote:
        print("Applying SMOTE oversampling...")
        X_train_prep = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_prep, y_train)
        print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts(normalize=True)}")
    
    # Optimize hyperparameters with Bayesian optimization if specified
    if optimize_hyperparams:
        print("Optimizing hyperparameters with Bayesian search...")
        
        # Define the hyperparameter search space
        param_space = {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(5, 30),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2']),
            'classifier__class_weight': Categorical([None, 'balanced', 'balanced_subsample'])
        }
        
        # Create the Bayesian search CV
        bayes_search = BayesSearchCV(
            model_pipeline,
            param_space,
            n_iter=20,
            cv=5,
            scoring='f1',
            random_state=42
        )
        
        # Fit the Bayesian search to find optimal parameters
        if use_smote:
            bayes_search.fit(X_train, y_train, 
                            classifier__sample_weight=np.ones(len(y_train)))
        else:
            bayes_search.fit(X_train, y_train)
        
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best CV score: {bayes_search.best_score_:.4f}")
        
        # Return the best model
        best_model = bayes_search.best_estimator_
        
    else:
        # Train with default hyperparameters
        if use_smote:
            # Apply SMOTE and fit
            model_pipeline.fit(X_train, y_train)
        else:
            # Fit without SMOTE
            model_pipeline.fit(X_train, y_train)
        
        best_model = model_pipeline
    
    return best_model

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label='Random Forest')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png')
    
    # Feature importance analysis
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Get categorical feature names after one-hot encoding
        cat_features = preprocessor.transformers_[1][2]
        cat_transformer = preprocessor.transformers_[1][1]
        cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(cat_features)
        
        # Get numerical feature names
        num_features = preprocessor.transformers_[0][2]
        
        # Combine all feature names
        feature_names = list(num_features) + list(cat_feature_names)
        
        # Extract feature importances
        importances = classifier.feature_importances_
        
        # Create sorted feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # SHAP analysis for interpretability
        try:
            # Create a SHAP explainer
            X_test_processed = preprocessor.transform(X_test)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_processed)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values[1], X_test_processed, feature_names=feature_names,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            
            # Plot SHAP dependency plots for top features
            for i in range(min(3, len(feature_importance))):
                feature = feature_importance.iloc[i]['Feature']
                feature_idx = feature_names.index(feature)
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature_idx, shap_values[1], X_test_processed, 
                                    feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f'shap_dependence_{i}.png')
                
        except Exception as e:
            print(f"SHAP analysis error: {e}")
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

# 5. GIS Visualization
def create_gis_visualizations(df, gis_data):
    """
    Create GIS visualizations of fitness data
    """
    # Create a map centered on Hong Kong
    m = folium.Map(location=[22.3, 114.17], zoom_start=11)
    
    # Add fitness tracks to the map
    for _, track in gis_data.iterrows():
        points = track['coords']
        folium.PolyLine(
            points,
            color='green',
            weight=5,
            tooltip=f"{track['name']} ({track['length_km']}km)"
        ).add_to(m)
    
    # Create a heatmap of adherence
    adherence_data = df[['latitude', 'longitude', 'adherence_probability']].values.tolist()
    HeatMap(adherence_data, radius=15).add_to(m)
    
    # Add markers for high and low adherence districts
    district_adherence = df.groupby('district')['adherence'].mean().reset_index()
    district_coords = df.groupby('district')[['latitude', 'longitude']].mean().reset_index()
    
    merged_districts = pd.merge(district_adherence, district_coords, on='district')
    
    for _, district in merged_districts.iterrows():
        color = 'green' if district['adherence'] > 0.5 else 'red'
        folium.CircleMarker(
            location=[district['latitude'], district['longitude']],
            radius=district['adherence'] * 20,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{district['district']}: {district['adherence']:.2f} adherence rate"
        ).add_to(m)
    
    # Save the map
    m.save('fitness_adherence_map.html')
    
    # Create district-level comparison chart
    plt.figure(figsize=(14, 8))
    district_adherence = district_adherence.sort_values('adherence', ascending=False)
    
    colors = ['green' if x > 0.5 else 'red' for x in district_adherence['adherence']]
    
    sns.barplot(x='adherence', y='district', data=district_adherence, palette=colors)
    plt.title('Fitness Program Adherence by District')
    plt.xlabel('Adherence Rate')
    plt.ylabel('District')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('adherence_by_district.png')
    
    return m

# 6. Deployment Functions
def create_api_endpoint(model):
    """
    Simulation of API endpoint creation for model serving
    In a real implementation, this would use Flask or FastAPI
    """
    print("Creating API endpoint for model deployment...")
    
    # Define a prediction function that the API would use
    def predict_adherence(user_data):
        """
        Sample prediction function
        
        Parameters:
        user_data (dict): User information including:
            - age_group: string
            - gender: string
            - district: string
            - exercise_frequency: string
            - exercise_type: string
            - muscular_strength: float
            - bmi: float
            - latitude: float
            - longitude: float
        
        Returns:
        dict: Prediction results
        """
        # Convert single record to DataFrame
        input_df = pd.DataFrame([user_data])
        
        # Engineer features (simplified)
        age_mapping = {
            '7-12': 10, '13-19': 16, '20-39': 30, '40-59': 50, '60-79': 70
        }
        input_df['age_numeric'] = input_df['age_group'].map(age_mapping)
        
        freq_mapping = {
            'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Regularly': 3, 'Daily': 4
        }
        input_df['exercise_frequency_numeric'] = input_df['exercise_frequency'].map(freq_mapping)
        
        # Other feature engineering steps...
        
        # Make prediction
        adherence_prob = model.predict_proba(input_df)[0, 1]
        adherence_class = 1 if adherence_prob > 0.5 else 0
        
        return {
            'adherence_probability': float(adherence_prob),
            'adherence_class': int(adherence_class),
            'recommendation': get_recommendation(user_data, adherence_prob)
        }
    
    def get_recommendation(user_data, adherence_prob):
        """Generate a personalized recommendation"""
        if adherence_prob < 0.3:
            level = "low"
            if user_data['exercise_frequency'] in ['Never', 'Rarely']:
                activity = "start with short 10-minute walks"
            else:
                activity = "try group fitness classes for motivation"
        elif adherence_prob < 0.7:
            level = "moderate"
            activity = f"consider {user_data['exercise_type']} 2-3 times per week"
        else:
            level = "high"
            activity = "maintain your routine with progressive challenges"
            
        return f"Based on your profile, you have a {level} probability of adhering to a fitness program. We recommend you {activity}."
    
    # Example API Usage (simulation)
    sample_user = {
        'age_group': '20-39',
        'gender': 'F',
        'district': 'Central & Western',
        'exercise_frequency': 'Sometimes',
        'exercise_type': 'Aerobic',
        'muscular_strength': 65.0,
        'bmi': 23.5,
        'latitude': 22.2826,
        'longitude': 114.1452
    }
    
    prediction = predict_adherence(sample_user)
    print(f"API Endpoint Example:")
    print(f"Input: {sample_user}")
    print(f"Output: {prediction}")
    
    return {
        'predict_function': predict_adherence,
        'sample_input': sample_user,
        'sample_output': prediction
    }

def create_dashboard(df, evaluation_metrics):
    """
    Create a dashboard for policymakers
    Simulates a Dash application
    """
    print("Creating dashboard for policymakers...")
    
    # In a real implementation, this would be a Dash app
    # Here we'll just create and save the visualizations
    
    # 1. District Adherence Map (already created in GIS visualization)
    
    # 2. Demographic Breakdown
    plt.figure(figsize=(14, 10))
    
    # Age group adherence
    plt.subplot(2, 2, 1)
    age_adherence = df.groupby('age_group')['adherence'].mean().reset_index()
    sns.barplot(x='age_group', y='adherence', data=age_adherence)
    plt.title('Adherence by Age Group')
    plt.ylim(0, 1)
    
    # Gender adherence
    plt.subplot(2, 2, 2)
    gender_adherence = df.groupby('gender')['adherence'].mean().reset_index()
    sns.barplot(x='gender', y='adherence', data=gender_adherence)
    plt.title('Adherence by Gender')
    plt.ylim(0, 1)
    
    # Income level adherence
    plt.subplot(2, 2, 3)
    income_adherence = df.groupby('income_level')['adherence'].mean().reset_index()
    income_adherence['income_level'] = pd.Categorical(income_adherence['income_level'], 
                                                    categories=['Low', 'Medium', 'High'])
    income_adherence = income_adherence.sort_values('income_level')
    sns.barplot(x='income_level', y='adherence', data=income_adherence)
    plt.title('Adherence by Income Level')
    plt.ylim(0, 1)
    
    # Exercise frequency adherence
    plt.subplot(2, 2, 4)
    freq_adherence = df.groupby('exercise_frequency')['adherence'].mean().reset_index()
    freq_order = ['Never', 'Rarely', 'Sometimes', '# Hong Kong Fitness Program Adherence Prediction Model
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
import folium
from folium.plugins import HeatMap
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection and Preprocessing
def load_fitness_survey_data(filepath):
    """
    Load and preprocess LCSD fitness survey data
    """
    # Simulating data loading as we don't have actual data files
    print(f"Loading data from {filepath}")
    
    # Create synthetic dataset based on survey information in 1fit.md
    np.random.seed(42)
    n_samples = 9300  # Sample size mentioned in the document
    
    # Demographic features
    age_groups = ['7-12', '13-19', '20-39', '40-59', '60-79']
    age_weights = [0.15, 0.15, 0.3, 0.25, 0.15]  # Distribution weights
    ages = np.random.choice(age_groups, size=n_samples, p=age_weights)
    
    gender = np.random.choice(['M', 'F'], size=n_samples)
    
    # 18 districts in Hong Kong
    districts = [
        'Central & Western', 'Wan Chai', 'Eastern', 'Southern', 
        'Yau Tsim Mong', 'Sham Shui Po', 'Kowloon City', 'Wong Tai Sin', 
        'Kwun Tong', 'Kwai Tsing', 'Tsuen Wan', 'Tuen Mun', 
        'Yuen Long', 'North', 'Tai Po', 'Sha Tin', 
        'Sai Kung', 'Islands'
    ]
    # Some districts have higher population density
    district_weights = [
        0.05, 0.05, 0.07, 0.05, 0.08, 0.07, 0.06, 0.06, 
        0.08, 0.05, 0.05, 0.06, 0.06, 0.04, 0.04, 0.07, 
        0.04, 0.02
    ]
    districts_sampled = np.random.choice(districts, size=n_samples, p=district_weights)
    
    # Income levels (for considering bias mitigation)
    income_levels = ['Low', 'Medium', 'High']
    # Higher income in certain districts
    income_prob = []
    for district in districts_sampled:
        if district in ['Central & Western', 'Wan Chai', 'Southern']:
            income_prob.append([0.2, 0.3, 0.5])  # Higher probability of high income
        elif district in ['Sham Shui Po', 'Wong Tai Sin', 'Kwun Tong']:
            income_prob.append([0.5, 0.3, 0.2])  # Higher probability of low income
        else:
            income_prob.append([0.3, 0.4, 0.3])  # Balanced
    
    income = [np.random.choice(income_levels, p=prob) for prob in income_prob]
    
    # Behavioral features
    exercise_frequency = []
    for age_group in ages:
        if age_group in ['7-12', '13-19']:
            # Children have varied patterns due to school
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.2, 0.3, 0.2, 0.2, 0.1]
            )
        elif age_group in ['20-39']:
            # Young adults cite "lack of time"
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.3, 0.3, 0.2, 0.15, 0.05]
            )
        elif age_group in ['40-59']:
            # Middle-aged adults
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.25, 0.35, 0.2, 0.15, 0.05]
            )
        else:
            # Older adults might have more time but less mobility
            freq = np.random.choice(
                ['Never', 'Rarely', 'Sometimes', 'Regularly', 'Daily'],
                p=[0.3, 0.2, 0.2, 0.2, 0.1]
            )
        exercise_frequency.append(freq)
    
    # Exercise types
    exercise_types = []
    for gender_val, age_group in zip(gender, ages):
        type_probs = []
        if age_group in ['7-12', '13-19']:
            # Children more likely to do sports
            type_probs = [0.4, 0.2, 0.2, 0.1, 0.1]
        elif gender_val == 'M':
            # Males more likely to do strength training
            type_probs = [0.2, 0.3, 0.15, 0.25, 0.1]
        else:
            # Females more likely to do aerobic/yoga
            type_probs = [0.2, 0.15, 0.3, 0.15, 0.2]
        
        exercise_types.append(np.random.choice(
            ['Sports', 'Strength', 'Aerobic', 'Walking/Jogging', 'Yoga/Flexibility'],
            p=type_probs
        ))
    
    # Physiological features
    muscular_strength = []
    for age_group, gender_val, freq in zip(ages, gender, exercise_frequency):
        base_score = {
            '7-12': 60,
            '13-19': 65,
            '20-39': 70,
            '40-59': 65,
            '60-79': 50
        }[age_group]
        
        # Gender adjustment (slight difference in baseline strength)
        if gender_val == 'M':
            base_score += 5
        
        # Exercise frequency adjustment
        freq_adjustment = {
            'Never': -15,
            'Rarely': -5,
            'Sometimes': 0,
            'Regularly': 10,
            'Daily': 15
        }[freq]
        
        # Add some noise
        final_score = base_score + freq_adjustment + np.random.normal(0, 8)
        
        # Clamp between 0 and 100
        muscular_strength.append(max(0, min(100, final_score)))
    
    # BMI values (simulate obesity markers)
    bmi_values = []
    for age_group, freq in zip(ages, exercise_frequency):
        if freq in ['Never', 'Rarely']:
            base_bmi = 26 + np.random.normal(0, 3)  # Higher baseline for sedentary
        else:
            base_bmi = 22 + np.random.normal(0, 2)  # Lower baseline for active
            
        # Age adjustments
        if age_group in ['7-12']:
            base_bmi -= 4  # Children typically have lower BMI
        elif age_group in ['60-79']:
            base_bmi += 1  # Older adults often have slightly higher BMI
            
        bmi_values.append(max(15, min(40, base_bmi)))  # Clamp to realistic range
    
    # Create synthetic coordinates for each district for GIS integration
    # These are approximate center points for Hong Kong districts
    district_coords = {
        'Central & Western': (22.2826, 114.1452),
        'Wan Chai': (22.2808, 114.1826),
        'Eastern': (22.2845, 114.2256),
        'Southern': (22.2458, 114.1600),
        'Yau Tsim Mong': (22.3203, 114.1694),
        'Sham Shui Po': (22.3303, 114.1622),
        'Kowloon City': (22.3287, 114.1839),
        'Wong Tai Sin': (22.3419, 114.1953),
        'Kwun Tong': (22.3100, 114.2260),
        'Kwai Tsing': (22.3561, 114.1324),
        'Tsuen Wan': (22.3725, 114.1170),
        'Tuen Mun': (22.3908, 113.9725),
        'Yuen Long': (22.4445, 114.0225),
        'North': (22.4940, 114.1386),
        'Tai Po': (22.4513, 114.1644),
        'Sha Tin': (22.3864, 114.1928),
        'Sai Kung': (22.3809, 114.2707),
        'Islands': (22.2627, 113.9456)
    }
    
    # Add some noise to coordinates to simulate addresses in each district
    lats = []
    longs = []
    for district in districts_sampled:
        base_lat, base_long = district_coords[district]
        # Add noise (approximately within a 1-2km radius)
        lat = base_lat + np.random.normal(0, 0.01)
        long = base_long + np.random.normal(0, 0.01)
        lats.append(lat)
        longs.append(long)
    
    # Synthesize fitness track data
    fitness_tracks = [
        {'name': 'Victoria Park Jogging Trail', 'coords': [(22.2824, 114.1884), (22.2836, 114.1892)]},
        {'name': 'Kowloon Park Fitness Trail', 'coords': [(22.3023, 114.1693), (22.3045, 114.1698)]},
        {'name': 'Sha Tin Shing Mun River Track', 'coords': [(22.3794, 114.1878), (22.3825, 114.1901)]},
        {'name': 'Tseung Kwan O Waterfront Track', 'coords': [(22.3067, 114.2608), (22.3102, 114.2638)]},
        {'name': 'Tai Po Waterfront Park Track', 'coords': [(22.4502, 114.1711), (22.4530, 114.1732)]},
    ]
    
    # Calculate distance to nearest fitness track
    distances_to_track = []
    for lat, long in zip(lats, longs):
        min_distance = float('inf')
        for track in fitness_tracks:
            for track_coord in track['coords']:
                dist = haversine((lat, long), track_coord)
                min_distance = min(min_distance, dist)
        distances_to_track.append(min_distance)
    
    # Target variable: program adherence 
    # Factors that increase adherence probability:
    # - Regular exercise frequency
    # - Younger adults and middle-aged (20-59)
    # - Proximity to fitness tracks
    # - Better muscular strength scores
    
    adherence_prob = []
    for freq, age, dist, strength, bmi in zip(
        exercise_frequency, ages, distances_to_track, muscular_strength, bmi_values
    ):
        # Base probability
        prob = 0.5
        
        # Exercise frequency impact
        freq_factor = {
            'Never': -0.25,
            'Rarely': -0.15,
            'Sometimes': 0,
            'Regularly': 0.15,
            'Daily': 0.25
        }[freq]
        prob += freq_factor
        
        # Age impact
        if age in ['20-39', '40-59']:
            prob += 0.05
        elif age in ['7-12']:
            prob += 0.1  # Children often follow programs in school
        elif age in ['60-79']:
            prob -= 0.05  # Older adults might face mobility issues
            
        # Distance impact - closer is better
        if dist < 2:  # Within 2km
            prob += 0.1
        elif dist > 5:  # More than 5km away
            prob -= 0.1
            
        # Fitness level impact
        if strength > 70:
            prob += 0.05  # Already fit people might stick with programs
        elif strength < 40:
            prob -= 0.05  # Very unfit people might give up easier
            
        # BMI impact
        if bmi > 30:  # Obese
            prob -= 0.05  # Might face more challenges
            
        # Add some randomness
        prob += np.random.normal(0, 0.1)
        
        # Clamp between 0 and 1
        prob = max(0, min(1, prob))
        adherence_prob.append(prob)
    
    # Binary outcome based on probability
    adherence = [1 if p > 0.5 else 0 for p in adherence_prob]
    
    # Create the dataframe
    data = pd.DataFrame({
        'age_group': ages,
        'gender': gender,
        'district': districts_sampled,
        'income_level': income,
        'exercise_frequency': exercise_frequency,
        'exercise_type': exercise_types,
        'muscular_strength': muscular_strength,
        'bmi': bmi_values,
        'latitude': lats,
        'longitude': longs,
        'distance_to_track': distances_to_track,
        'adherence_probability': adherence_prob,
        'adherence': adherence
    })
    
    print(f"Created synthetic dataset with {len(data)} samples")
    print(f"Adherence rate: {data['adherence'].mean():.2f}")
    
    return data

def load_gis_data():
    """
    Load GIS data for fitness tracks
    In a real implementation, this would load from GIS files
    """
    # For simplicity, we'll create synthetic GIS data
    # In a real implementation, you'd use geopandas to load shapefiles
    
    fitness_tracks = [
        {'name': 'Victoria Park Jogging Trail', 
         'length_km': 1.2, 
         'elevation_gain': 5,
         'coords': [(22.2824, 114.1884), (22.2836, 114.1892)]},
        {'name': 'Kowloon Park Fitness Trail', 
         'length_km': 0.8, 
         'elevation_gain': 15,
         'coords': [(22.3023, 114.1693), (22.3045, 114.1698)]},
        {'name': 'Sha Tin Shing Mun River Track', 
         'length_km': 2.5, 
         'elevation_gain': 0,
         'coords': [(22.3794, 114.1878), (22.3825, 114.1901)]},
        {'name': 'Tseung Kwan O Waterfront Track', 
         'length_km': 3.0, 
         'elevation_gain': 5,
         'coords': [(22.3067, 114.2608), (22.3102, 114.2638)]},
        {'name': 'Tai Po Waterfront Park Track', 
         'length_km': 1.5, 
         'elevation_gain': 10,
         'coords': [(22.4502, 114.1711), (22.4530, 114.1732)]},
    ]
    
    return pd.DataFrame(fitness_tracks)

# 2. Feature Engineering
def engineer_features(df, gis_data):
    """
    Create derived features from raw data
    """
    # Create age as numeric feature
    age_mapping = {
        '7-12': 10,
        '13-19': 16,
        '20-39': 30,
        '40-59': 50,
        '60-79': 70
    }
    df['age_numeric'] = df['age_group'].map(age_mapping)
    
    # Create exercise frequency as numeric
    freq_mapping = {
        'Never': 0,
        'Rarely': 1,
        'Sometimes': 2,
        'Regularly': 3,
        'Daily': 4
    }
    df['exercise_frequency_numeric'] = df['exercise_frequency'].map(freq_mapping)
    
    # Create income level as numeric
    income_mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    df['income_numeric'] = df['income_level'].map(income_mapping)
    
    # Create BMI category
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    
    # Distance to track buckets
    def distance_category(dist):
        if dist < 1:
            return 'Very Close (<1km)'
        elif dist < 3:
            return 'Close (1-3km)'
        elif dist < 5:
            return 'Moderate (3-5km)'
        else:
            return 'Far (>5km)'
    
    df['distance_category'] = df['distance_to_track'].apply(distance_category)
    
    # Muscular strength categories
    def strength_category(score):
        if score < 40:
            return 'Low'
        elif score < 70:
            return 'Medium'
        else:
            return 'High'
    
    df['strength_category'] = df['muscular_strength'].apply(strength_category)
    
    # Create interaction features
    df['age_exercise_interaction'] = df['age_numeric'] * df['exercise_frequency_numeric']
    df['age_bmi_interaction'] = df['age_numeric'] * df['bmi']
    
    return df

# 3. Model Building
def build_model(X_train, y_train, use_smote=True, optimize_hyperparams=True):
    """
    Build and train the random forest model
    """
    # Define categorical and numerical features
    categorical_features = [
        'age_group', 'gender', 'district', 'income_level', 
        'exercise_frequency', 'exercise_type', 'bmi_category',
        'distance_category', 'strength_category'
    ]
    
    numerical_features = [
        'muscular_strength', 'bmi', 'distance_to_track',
        'age_numeric', 'exercise_frequency_numeric', 'income_numeric',
        'age_exercise_interaction', 'age_bmi_interaction'
    ]
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Handle class imbalance with SMOTE if specified
    if use_smote:
        print("Applying SMOTE oversampling...")
        X_train_prep = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_prep, y_train)
        print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts(normalize=True)}")
    
    # Optimize hyperparameters with Bayesian optimization if specified
    if optimize_hyperparams:
        print("Optimizing hyperparameters with Bayesian search...")
        
        # Define the hyperparameter search space
        param_space = {
            'classifier__n_estimators': Integer(50, 300),
            'classifier__max_depth': Integer(5, 30),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2']),
            'classifier__class_weight': Categorical([None, 'balanced', 'balanced_subsample'])
        }
        
        # Create the Bayesian search CV
        bayes_search = BayesSearchCV(
            model_pipeline,
            param_space,
            n_iter=20,
            cv=5,
            scoring='f1',
            random_state=42
        )
        
        # Fit the Bayesian search to find optimal parameters
        if use_smote:
            bayes_search.fit(X_train, y_train, 
                            classifier__sample_weight=np.ones(len(y_train)))
        else:
            bayes_search.fit(X_train, y_train)
        
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best CV score: {bayes_search.best_score_:.4f}")
        
        # Return the best model
        best_model = bayes_search.best_estimator_
        
    else:
        # Train with default hyperparameters
        if use_smote:
            # Apply SMOTE and fit
            model_pipeline.fit(X_train, y_train)
        else:
            # Fit without SMOTE
            model_pipeline.fit(X_train, y_train)
        
        best_model = model_pipeline
    
    return best_model

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label='Random Forest')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png')
    
    # Feature importance analysis
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Get categorical feature names after one-hot encoding
        cat_features = preprocessor.transformers_[1][2]
        cat_transformer = preprocessor.transformers_[1][1]
        cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(cat_features)
        
        # Get numerical feature names
        num_features = preprocessor.transformers_[0][2]
        
        # Combine all feature names
        feature_names = list(num_features) + list(cat_feature_names)
        
        # Extract feature importances
        importances = classifier.feature_importances_
        
        # Create sorted feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # SHAP analysis for interpretability
        try:
            # Create a SHAP explainer
            X_test_processed = preprocessor.transform(X_test)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_processed)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values[1], X_test_processed, feature_names=feature_names,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
            
            # Plot SHAP dependency plots for top features
            for i in range(min(3, len(feature_importance))):
                feature = feature_importance.iloc[i]['Feature']
                feature_idx = feature_names.index(feature)
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature_idx, shap_values[1], X_test_processed, 
                                    feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f'shap_dependence_{i}.png')
                
        except Exception as e:
            print(f"SHAP analysis error: {e}")
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

# 5. GIS Visualization
def create_gis_visualizations(df, gis_data):
    """
    Create GIS visualizations of fitness data
    """
    # Create a map centered on Hong Kong
    m = folium.Map(location=[22.3, 114.17], zoom_start=11)
    
    # Add fitness tracks to the map
    for _, track in gis_data.iterrows():
        points = track['coords']
        folium.PolyLine(
            points,
            color='green',
            weight=5,
            tooltip=f"{track['name']} ({track['length_km']}km)"
        ).add_to(m)
    
    # Create a heatmap of adherence
    adherence_data = df[['latitude', 'longitude', 'adherence_probability']].values.tolist()
    HeatMap(adherence_data, radius=15).add_to(m)
    
    # Add markers for high and low adherence districts
    district_adherence = df.groupby('district')['adherence'].mean().reset_index()
    district_coords = df.groupby('district')[['latitude', 'longitude']].mean().reset_index()
    
    merged_districts = pd.merge(district_adherence, district_coords, on='district')
    
    for _, district in merged_districts.iterrows():
        color = 'green' if district['adherence'] > 0.5 else 'red'
        folium.CircleMarker(
            location=[district['latitude'], district['longitude']],
            radius=district['adherence'] * 20,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{district['district']}: {district['adherence']:.2f} adherence rate"
        ).add_to(m)
    
    # Save the map
    m.save('fitness_adherence_map.html')
    
    # Create district-level comparison chart
    plt.figure(figsize=(14, 8))
    district_adherence = district_adherence.sort_values('adherence', ascending=False)
    
    colors = ['green' if x > 0.5 else 'red' for x in district_adherence['adherence']]
    
    sns.barplot(x='adherence', y='district', data=district_adherence, palette=colors)
    plt.title('Fitness Program Adherence by District')
    plt.xlabel('Adherence Rate')
    plt.ylabel('District')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('adherence_by_district.png')
    
    return m

# 6. Deployment Functions
def create_api_endpoint(model):
    """
    Simulation of API endpoint creation for model serving
    In a real implementation, this would use Flask or FastAPI
    """
    print("Creating API endpoint for model deployment...")
    
    # Define a prediction function that the API would use
    def predict_adherence(user_data):
        """
        Sample prediction function
        
        Parameters:
        user_data (dict): User information including:
            - age_group: string
            - gender: string
            - district: string
            - exercise_frequency: string
            - exercise_type: string
            - muscular_strength: float
            - bmi: float
            - latitude: float
            - longitude: float
        
        Returns:
        dict: Prediction results
        """
        # Convert single record to DataFrame
        input_df = pd.DataFrame([user_data])
        
        # Engineer features (simplified)
        age_mapping = {
            '7-12': 10, '13-19': 16, '20-39': 30, '40-59': 50, '60-79': 70
        }
        input_df['age_numeric'] = input_df['age_group'].map(age_mapping)
        
        freq_mapping = {
            'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Regularly': 3, 'Daily': 4
        }
        input_df['exercise_frequency_numeric'] = input_df['exercise_frequency'].map(freq_mapping)
        
        # Other feature engineering steps...
        
        # Make prediction
        adherence_prob = model.predict_proba(input_df)[0, 1]
        adherence_class = 1 if adherence_prob > 0.5 else 0
        
        return {
            'adherence_probability': float(adherence_prob),
            'adherence_class': int(adherence_class),
            'recommendation': get_recommendation(user_data, adherence_prob)
        }
    
    def get_recommendation(user_data, adherence_prob):
        """Generate a personalized recommendation"""
        if adherence_prob < 0.3:
            level = "low"
            if user_data['exercise_frequency'] in ['Never', 'Rarely']:
                activity = "start with short 10-minute walks"
            else:
                activity = "try group fitness classes for motivation"
        elif adherence_prob < 0.7:
            level = "moderate"
            activity = f"consider {user_data['exercise_type']} 2-3 times per week"
        else:
            level = "high"
            activity = "maintain your routine with progressive challenges"
            
        return f"Based on your profile, you have a {level} probability of adhering to a fitness program. We recommend you {activity}."
    
    # Example API Usage (simulation)
    sample_user = {
        'age_group': '20-39',
        'gender': 'F',
        'district': 'Central & Western',
        'exercise_frequency': 'Sometimes',
        'exercise_type': 'Aerobic',
        'muscular_strength': 65.0,
        'bmi': 23.5,
        'latitude': 22.2826,
        'longitude': 114.1452
    }
    
    prediction = predict_adherence(sample_user)
    print(f"API Endpoint Example:")
    print(f"Input: {sample_user}")
    print(f"Output: {prediction}")
    
    return {
        'predict_function': predict_adherence,
        'sample_input': sample_user,
        'sample_output': prediction
    }

def create_dashboard(df, evaluation_metrics):
    """
    Create a dashboard for policymakers
    Simulates a Dash application
    """
    print("Creating dashboard for policymakers...")
    
    # In a real implementation, this would be a Dash app
    # Here we'll just create and save the visualizations
    
    # 1. District Adherence Map (already created in GIS visualization)
    
    # 2. Demographic Breakdown
    plt.figure(figsize=(14, 10))
    
    # Age group adherence
    plt.subplot(2, 2, 1)
    age_adherence = df.groupby('age_group')['adherence'].mean().reset_index()
    sns.barplot(x='age_group', y='adherence', data=age_adherence)
    plt.title('Adherence by Age Group')
    plt.ylim(0, 1)
    
    # Gender adherence
    plt.subplot(2, 2, 2)
    gender_adherence = df.groupby('gender')['adherence'].mean().reset_index()
    sns.barplot(x='gender', y='adherence', data=gender_adherence)
    plt.title('Adherence by Gender')
    plt.ylim(0, 1)
    
    # Income level adherence
    plt.subplot(2, 2, 3)
    income_adherence = df.groupby('income_level')['adherence'].mean().reset_index()
    income_adherence['income_level'] = pd.Categorical(income_adherence['income_level'], 
                                                    categories=['Low', 'Medium', 'High'])
    income_adherence = income_adherence.sort_values('income_level')
    sns.barplot(x='income_level', y='adherence', data=income_adherence)
    plt.title('Adherence by Income Level')
    plt.ylim(0, 1)
    
    # Exercise frequency adherence
    plt.subplot(2, 2, 4)
    freq_adherence = df.groupby('exercise_frequency')['adherence'].mean().reset_index()
    freq_order = ['Never', 'Rarely', 'Sometimes', '