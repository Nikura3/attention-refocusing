import os
from transformers import CLIPModel, CLIPProcessor

os.makedirs("clip", exist_ok=True)

CLIPModel.from_pretrained("openai/clip-vit-large-patch14").save_pretrained("clip/openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").save_pretrained("clip/openai/clip-vit-large-patch14")