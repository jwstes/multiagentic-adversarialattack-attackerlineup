from transformers import pipeline
import torch
import json
import time

from misc import attackMethods

prompt = open('misc/prompt-thinkingAgent.txt', 'r').read()
# analysis = json.loads(open('output/analysis.json', 'r').read())
# context = open('output/vectorStoreResults.txt', 'r').read()

def formatAttackMethods():
    stringifiedMethods = ""
    for method in attackMethods.all_methods:
        stringifiedMethods += f"{method}, "

    return stringifiedMethods

def thinkAndSelectMethod(imageAnalysis, context):
    print("[Thinking Agent]: Thinking...")

    sT = time.time()
    
    model_id = "openai/gpt-oss-20b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"""
    Available methods: {formatAttackMethods()}
    Image facts: {imageAnalysis}
    Retrieved evidence: {context}
    """},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=1024,
    )

    eT = time.time()

    print("[Thinking Agent]: Done. It took (seconds)", eT - sT)
    finalOutput = outputs[0]["generated_text"][-1]['content']

    file = open('output/thinkingAgentOutput.txt', 'w')
    file.writelines(finalOutput)
    file.close()

    return finalOutput

