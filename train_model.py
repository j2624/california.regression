from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
# 1) Load California Housing dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2) Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Create a pipeline: scaler + model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

# 4) Train model
pipe.fit(X_train, y_train)

# 5) Evaluate model
pred = pipe.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("R²:", round(r2_score(y_test, pred), 3))


# 6) Save model
joblib.dump(pipe, "california_model.joblib")
print(" Model saved as california_model.joblib")