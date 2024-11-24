import coremltools as ct
import pickle

# Load your trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Define input and output feature names
input_features = [f'feature_{i}' for i in range(42)]  # Assuming 42 features
output_feature = 'predicted_character'

# Convert to Core ML model
coreml_model = ct.converters.sklearn.convert(model, input_features, output_feature)

# Save the Core ML model
coreml_model.save('ASLClassifier.mlmodel')

# Load the Core ML model
coreml_model = ct.models.MLModel("ASLClassifier.mlmodel")

# Add metadata
coreml_model.author = "Your Name"
coreml_model.short_description = "ASL Classifier using RandomForest"
coreml_model.input_description["feature_0"] = "Normalized x and y coordinates of landmarks"
coreml_model.output_description["predicted_character"] = "Predicted ASL character"

# Save the updated model
coreml_model.save("ASLClassifier_with_metadata.mlmodel")
# Load the Core ML model
coreml_model = ct.models.MLModel("ASLClassifier_with_metadata.mlmodel")

# Validate the model
validation_result = coreml_model.get_spec()
print("Validation Results:", validation_result)

# Optionally, inspect the model's spec
print(coreml_model.get_spec())