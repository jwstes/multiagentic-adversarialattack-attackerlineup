from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch
import config

model_name = f"{config.hfModelFamily}{config.hfModelName}"
save_path = f"./{config.hfModelName}-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set False for 8-bit
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)