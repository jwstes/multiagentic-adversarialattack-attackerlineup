import warnings
from transformers import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from misc import config
import torch
import time


model_name = f"{config.hfModelFamily}{config.hfModelName}"
min_pixels = config.min_pixels
max_pixels = config.max_pixels
MAX_TOKEN_COUNT = config.ia_max_token_count

prompt = open('misc/prompt-infoAgent.txt', 'r').read()

def init():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Set False for 8-bit
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    return model, processor

def gatherInformation(areasOfInterest, model, processor, imagePath):
    analysis = []
    for interest in areasOfInterest:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": imagePath,
                    },
                    {
                        "type": "text", 
                        "text": f"{prompt} \n\n {interest}",
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKEN_COUNT)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        modelResponse = output_text[0]
        tailAndHeaderSrip = modelResponse.split('```json')[1].split('```')[0]
        jsonVersion = json.loads(tailAndHeaderSrip)
        for item in jsonVersion:
            analysis.append(item)

        print(f"{interest} : Done")

    return analysis

def extractInfo(model, processor):
    print("[Info Agent]: Extracting Information...")
    sT = time.time()
    areasOfInterest = [
        "lighting, mismatched, around, unnatural, skin",
        "manipulation, artifacts, edges, inconsistencies, blending",
        "manipulated, features, tone, color, inconsistent",
        "forensic, definitive, tones, mismatches, transitions",
        "natural, consistent, hairline, body, jawline",
        "compositing, inspection, along, metadata, distortions",
        "hair, details, altered, neck, clues",
        "surrounding, background, video, jaw, head",
        # "strong, match, original, reference, subtle",
        # "unedited, indicators, frames, unaltered, visible",
        # "distorted, resolution, texture, mismatch, multiple",
        # "telltale, area, digitally, blend, overlaid",
        # "coherent, quality, detailed, areas, near",
        # "contours, conclude, overt, mouth, photograph",
        # "information, absolute"
    ]

    # model, processor = init()
    analysis = gatherInformation(areasOfInterest, model, processor, "images/fake.png")

    file = open("output/analysis.json", "w")
    file.write(json.dumps(analysis, indent=4))
    file.close()

    eT = time.time()
    print("[Info Agent]: Done. It took (seconds)", eT - sT)

    return analysis





# def extractInfo(model, processor, imagePath, imageName):
#     print("[Info Agent]: Extracting Information...")
#     sT = time.time()

#     areasOfInterest = [
#         "altered, area, around, artifacts, background",
#         # "blend, blending, body, clues, coherent",
#         # "color, compositing, conclude, consistent, contours",
#         # "definitive, detailed, digitally, distorted, edges",
#         # "features, forensic, hair, hairline, head",
#         # "inconsistencies, indicators, information, inspection, jaw",
#         # "jawline, lighting, manipulated, mismatched, mouth",
#         # "natural, near, neck, original, overlaid",
#         # "overt, photograph, quality, reference, resolution",
#         # "skin, strong, subtle, surrounding, telltale signs",
#         # "texture, tone, unaltered, unedited, unnatural, visible"
#     ]

#     # model, processor = init()
#     analysis = gatherInformation(areasOfInterest, model, processor, imagePath)

#     file = open(f"output/{imageName}.json", "w")
#     file.write(json.dumps(analysis, indent=4))
#     file.close()

#     eT = time.time()
#     print("[Info Agent]: Done. It took (seconds)", eT - sT)

#     return analysis

# def analysis():
#     from pathlib import Path
#     folder_path = Path("AADD-Dataset/hq")
#     image_filepaths = list(folder_path.glob("*.png"))
#     print(f"Found {len(image_filepaths)} .png images.")

#     model, processor = init()

#     for filepath in image_filepaths:
#         print("")
#         print("")
#         print("")
#         filename = filepath.stem
#         print(filepath, filename)
#         extractInfo(model, processor, str(filepath), filename)
#         print("")
#         print("")
#         print("")

#         break

# analysis()

