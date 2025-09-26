import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoProcessor, AutoModelForAudioClassification


pretrained_model_name_or_path = "facebook/w2v-bert-2.0"
# processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
model = AutoModelForAudioClassification.from_pretrained(pretrained_model_name_or_path)

print(model.config)
