import joblib
import torch

image_feature_extract_model = joblib.load("image_feature_extract.model")
print(image_feature_extract_model)
print(image_feature_extract_model(torch.ones(1,3,128,128)))