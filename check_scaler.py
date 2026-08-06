import joblib
target_scalers = joblib.load("models/svr_target_scalers.pkl")
print(type(target_scalers))
print(target_scalers)