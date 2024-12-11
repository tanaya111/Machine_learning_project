# model_.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class ModelBase:
    def load(self, data_path):
        """Load the dataset."""
        self.data = pd.read_csv(data_path)
        print(f"Data loaded from {data_path}")
        
    def preprocess(self):
        """Handle missing values, scale features, and split data."""
        self.data.fillna(self.data.mean(), inplace=True)  # Handle missing data by replacing with mean
        X = self.data.drop(columns=['target'])  # Replace 'target' with your actual target column name
        y = self.data['target']  # Replace 'target' with your actual target column name
        
        self.scaler = StandardScaler()  # Save scaler to reuse for predictions
        X_scaled = self.scaler.fit_transform(X)  # Fit and transform the training data
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    def train(self):
        """Train the model (to be overridden in subclasses)."""
        pass
    
    def test(self):
        """Test the model and print the evaluation report."""
        pass
    
    def predict(self, new_data):
        """Predict on new data."""
        new_data_scaled = self.scaler.transform(new_data)  # Apply the same scaling to new data
        predictions = self.model.predict(new_data_scaled)
        return predictions

class LogisticRegressionModel(ModelBase):
    def train(self):
        """Train the Logistic Regression model."""
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        print("Logistic Regression model trained.")
    
    def test(self):
        """Test the Logistic Regression model and print the evaluation report."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)
    
    def predict(self, new_data):
        """Predict using the Logistic Regression model."""
        return super().predict(new_data)  # Use the base class method for prediction

class RandomForestModel(ModelBase):
    def train(self):
        """Train the Random Forest model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Random Forest model trained.")
    
    def test(self):
        """Test the Random Forest model and print the evaluation report."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)
    
    def predict(self, new_data):
        """Predict using the Random Forest model."""
        return super().predict(new_data)  # Use the base class method for prediction
