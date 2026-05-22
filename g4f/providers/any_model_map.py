audio_models = []
image_models = []
vision_models = []
video_models = []

model_map = {
    "default": {"EncryptedProxy": "", "Chatai": ""},
    "gpt-4o": {"EncryptedProxy": "gpt-4o"},
    "gpt-4o-mini": {"EncryptedProxy": "gpt-4o-mini", "Chatai": "gpt-4o-mini"},
}

models_count = {}
parents = {}
model_aliases = {}
