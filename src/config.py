import os
from dotenv import load_dotenv

load_dotenv()

METHOD_PRINT_EVERY = int(os.getenv("METHOD_PRINT_EVERY", "3"))
STATUS_EVERY_SEC = float(os.getenv("STATUS_EVERY_SEC", "5.0"))


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://13.158.181.44:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY_KEY")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

MIXING_DESK_BACKEND = os.getenv("MIXING_DESK_BACKEND", "file")
MIXING_DESK_DIR = os.getenv("MIXING_DESK_DIR", "./data/mixing_desk")

MODELS_DIR = os.getenv("MODELS_DIR", "./models")

INFOAGENT_TEMPERATURE = float(os.getenv("INFOAGENT_TEMPERATURE", "0.2"))
INFOAGENT_MAX_TOKENS = int(os.getenv("INFOAGENT_MAX_TOKENS", "1500"))

# Default Conductor objectives
OBJ_TARGET_CLASS = int(os.getenv("OBJ_TARGET_CLASS", "0"))  # 0 => "real"
OBJ_TARGET_CONF = float(os.getenv("OBJ_TARGET_CONF", "0.90"))
OBJ_SSIM_MIN = float(os.getenv("OBJ_SSIM_MIN", "0.70"))
OBJ_EPS_MAX = float(os.getenv("OBJ_EPS_MAX", str(12/255)))  # L_inf bound in [0,1] scale
OBJ_AGGREGATE = os.getenv("OBJ_AGGREGATE", "mean")

# Device
DEVICE_CHOICE = os.getenv("DEVICE_CHOICE", "auto")  # "auto" or "cpu"



# Final Check settings
FINALCHECK_MODELS_DIR = os.getenv("FINALCHECK_MODELS_DIR", "./finalCheckModels")
FINALCHECK_NAME = os.getenv("FINALCHECK_NAME", "vit_b_16")  # currently supported: vit_b_16
FINALCHECK_RESIZE = int(os.getenv("FINALCHECK_RESIZE", "256"))       # pre-resize
FINALCHECK_CENTER_CROP = int(os.getenv("FINALCHECK_CENTER_CROP", "224"))  # center crop size
FINALCHECK_MIN_CONF = float(os.getenv("FINALCHECK_MIN_CONF", "0.0"))  # optional extra gate (0..1). If >0, require conf >= this too.


# Success policy (what triggers early stop)
# - "final_flip": success when ONLY the final check model changed prediction vs original (plus SSIM threshold)
# - "final_target": success when ONLY the final check model equals the target class (plus SSIM threshold)
# - "flip": (legacy) require all models to flip (surrogates + final)
SUCCESS_MODE = os.getenv("SUCCESS_MODE", "final_flip")  # "final_flip" | "final_target" | "flip"

# Optionally force a fixed target_class (e.g., 0 = "real"). -1 disables overriding the LLM.
FORCE_TARGET_CLASS = int(os.getenv("FORCE_TARGET_CLASS", "-1"))

# In legacy "flip" mode, you may also require surrogate avg_conf – leave false for final_* modes
REQUIRE_CONF_IN_FLIP = os.getenv("REQUIRE_CONF_IN_FLIP", "false").lower() == "true"


