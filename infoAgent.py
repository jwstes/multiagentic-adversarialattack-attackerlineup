from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import config

useQuantizedVersion = True
if useQuantizedVersion:
    model_name = f"./{config.hfModelName}-4bit"
else:
    model_name = f"{config.hfModelFamily}{config.hfModelName}"

min_pixels = 256*28*28
max_pixels = 1280*28*28
MAX_TOKEN_COUNT = 2048

prompt = open('prompt.txt', 'r').read()


def init():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
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

if __name__ == "__main__":
    areasOfInterest = [
        "lighting, mismatched, around, unnatural, skin",
        "manipulation, artifacts, edges, inconsistencies, blending",
        "manipulated, features, tone, color, inconsistent",
        "forensic, definitive, tones, mismatches, transitions",
        "natural, consistent, hairline, body, jawline",
        "compositing, inspection, along, metadata, distortions",
        "hair, details, altered, neck, clues",
        "surrounding, background, video, jaw, head",
        "strong, match, original, reference, subtle",
        "unedited, indicators, frames, unaltered, visible",
        "distorted, resolution, texture, mismatch, multiple",
        "telltale, area, digitally, blend, overlaid",
        "coherent, quality, detailed, areas, near",
        "contours, conclude, overt, mouth, photograph",
        "information, absolute"
    ]

    model, processor = init()
    analysis = gatherInformation(areasOfInterest, model, processor, "images/fake.png")





    file = open("analysis.json", "w")
    file.write(json.dumps(analysis, indent=4))
    file.close()


