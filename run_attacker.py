#!/usr/bin/env python3
import os
import sys
import json
import time
import traceback
from threading import Thread, Event

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.core.logging_setup import setup_logging
from src.core.mixing_desk import FileMixingDesk
from src.agents.conductor.agent import ConductorAgent
from src.agents.conductor.llm_conductor import ConductorLLMAgent
from src.agents.methods.fgsm_agent import FGSM_Agent
from src.agents.methods.pgd_agent import PGD_Agent
from src.agents.methods.advisors import MethodAdvisor
from src.agents.mixer.agent import MixerAgent
from src.agents.critique.surrogate_agent import SurrogateCritiqueAgent
from src.agents.critique.perceptual_agent import PerceptualCritiqueAgent
from src.agents.critique.finalcheck_agent import FinalCheckAgent
from src.agents.strategy.strategist_agent import StrategistAgent
from src.core.utils.images import file_sha1, pil_read_rgb, pil_to_tensor_unit
from src.config import (
    MODELS_DIR,
    DEVICE_CHOICE,
    FINALCHECK_NAME,
    SUCCESS_MODE,
    FORCE_TARGET_CLASS,
    REQUIRE_CONF_IN_FLIP,
    STATUS_EVERY_SEC
)

from src.core.vllm_client import VLLMClient
from src.agents.info_agent.agent import InfoAgent
from src.config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_ID, INFOAGENT_TEMPERATURE, INFOAGENT_MAX_TOKENS

from src.critique.detectors import load_classifier, get_device, build_transform_tensor
from src.finalcheck.model import load_final_model, build_spatial_transform_vit
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

logger = setup_logging()

def check_models():
    m1 = os.path.join(MODELS_DIR, "resnet50.pth")
    m2 = os.path.join(MODELS_DIR, "densenet121.pth")
    missing = [p for p in (m1, m2) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model weight files: {missing}")
    logger.info(f"Models found under {MODELS_DIR}")

def safe_thread(target, name: str, *args, **kwargs) -> Thread:
    def runner():
        try:
            target(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thread '{name}' crashed: {e}\n{traceback.format_exc()}")
    t = Thread(target=runner, name=name, daemon=True)
    t.start()
    return t

def compute_and_save_baseline(desk: FileMixingDesk, image_id: str):
    device = get_device(DEVICE_CHOICE)
    # Surrogates
    from pathlib import Path as P
    resnet = load_classifier("resnet50", P(MODELS_DIR) / "resnet50.pth", device)
    densenet = load_classifier("densenet121", P(MODELS_DIR) / "densenet121.pth", device)
    norm = build_transform_tensor()

    orig_path = desk.path_original(image_id)
    img = pil_read_rgb(orig_path, size=None)
    x = pil_to_tensor_unit(img).to(device)
    x_norm = norm(x.clone())

    with torch.no_grad():
        logits_r = resnet(x_norm)
        probs_r = F.softmax(logits_r, dim=1)[0].cpu().numpy().tolist()
        pred_r = int(logits_r.argmax(1).item())

        logits_d = densenet(x_norm)
        probs_d = F.softmax(logits_d, dim=1)[0].cpu().numpy().tolist()
        pred_d = int(logits_d.argmax(1).item())

    # Final check (vit)
    from src.config import FINALCHECK_MODELS_DIR, FINALCHECK_RESIZE, FINALCHECK_CENTER_CROP
    weights = Path(FINALCHECK_MODELS_DIR) / f"{FINALCHECK_NAME}.pth"
    final_model = load_final_model(FINALCHECK_NAME, weights, device)
    transform = build_spatial_transform_vit(FINALCHECK_RESIZE, FINALCHECK_CENTER_CROP)
    with torch.no_grad():
        t = transform(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
        logits_f = final_model(t)
        probs_f = F.softmax(logits_f, dim=1)[0].cpu().numpy().tolist()
        pred_f = int(logits_f.argmax(1).item())

    baseline = {
        "surrogates": {
            "resnet50": {"pred": pred_r, "probs": probs_r},
            "densenet121": {"pred": pred_d, "probs": probs_d},
        },
        "final": {
            FINALCHECK_NAME: {"pred": pred_f, "probs": probs_f}
        }
    }
    desk.save_baseline(image_id, baseline)
    logger.info(f"[Baseline] resnet50={pred_r} densenet121={pred_d} {FINALCHECK_NAME}={pred_f}")

def loop_method(agent, image_id: str, loops: int, interval: float, stop_event: Event):
    for i in range(loops):
        if stop_event.is_set():
            break
        agent.run_once(image_id, step=i)
        if stop_event.is_set():
            break
        time.sleep(interval)

def loop_mixer(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = MixerAgent(desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_resnet(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = SurrogateCritiqueAgent("resnet50", desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_densenet(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = SurrogateCritiqueAgent("densenet121", desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_critique_perceptual(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = PerceptualCritiqueAgent(desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_final_check(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    agent = FinalCheckAgent(desk)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def loop_strategist(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk, vllm: VLLMClient):
    agent = StrategistAgent(desk, vllm=vllm)
    while not stop_event.is_set():
        agent.run_once(image_id)
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def monitor_early_stop(image_id: str, desk: FileMixingDesk, stop_event: Event, interval: float = 2.0, no_early_stop: bool = False):
    from json import load
    obj_path = desk.path_objective(image_id)
    with open(obj_path, "r", encoding="utf-8") as f:
        obj = load(f)
    ssim_min = float(obj.get("ssim_min", 0.70))
    conf_target = float(obj.get("conf_target", 0.90))  # not used in final_* modes, kept for legacy

    start = time.time()
    last_print = 0.0
    while not stop_event.is_set():
        if (time.time() - start) > 900:
            logger.info("[Monitor] Max time exceeded; stopping.")
            stop_event.set()
            break

        # Read master metrics
        meta_path = os.path.join(desk.master_dir(image_id), "master_meta.json")
        avg_conf = 0.0
        ssim_val = 0.0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                metrics = meta.get("metrics", {}) or {}
                avg_conf = float(metrics.get("avg_conf") or 0.0)
                ssim_val = float(metrics.get("ssim") or 0.0)
            except Exception as e:
                logger.warning(f"[Monitor] Could not read master meta: {e}")

        # Panel (use lock-safe loader)
        panel = desk.load_feedback_panel(image_id)
        final_entry = None
        for e in panel.entries:
            if e.name.startswith("FinalCheck_"):
                final_entry = e.metrics or {}
                break

        final_flip = bool(final_entry.get("flipped", False)) if final_entry else False
        final_target_success = bool(final_entry.get("success_target", False)) if final_entry else False

        now = time.time()
        if now - last_print > 5.0:
            logger.info(
                f"[Monitor] ssim={ssim_val:.4f} avg_conf={avg_conf:.4f} "
                f"final_flip={final_flip} final_target={final_target_success} "
                f"mode={SUCCESS_MODE} (ssim_min={ssim_min:.2f})"
            )
            last_print = now

        # Success policy
        success = False
        if SUCCESS_MODE == "final_flip":
            success = (final_flip and ssim_val >= ssim_min)
        elif SUCCESS_MODE == "final_target":
            success = (final_target_success and ssim_val >= ssim_min)
        else:  # legacy "flip" across all models
            # For reference, legacy mode still supported:
            # require resnet & densenet flip + final flip, and optionally avg_conf threshold
            flips_ok = False
            res_flip = den_flip = False
            for e in panel.entries:
                if e.name == "SurrogateCritique_resnet50":
                    res_flip = bool((e.metrics or {}).get("flipped", False))
                elif e.name == "SurrogateCritique_densenet121":
                    den_flip = bool((e.metrics or {}).get("flipped", False))
            flips_ok = res_flip and den_flip and final_flip
            conf_ok = (avg_conf >= conf_target) if REQUIRE_CONF_IN_FLIP else True
            success = flips_ok and conf_ok and (ssim_val >= ssim_min)

        if not no_early_stop and success:
            logger.info("[Monitor] Success criteria met; stopping.")
            stop_event.set()
            break

        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)
    
    if not stop_event.is_set():
        # This path almost never hits; stop_event is set by guard.
        pass
    # After joining threads, check whether success occurred
    panel = desk.load_feedback_panel(image_id)
    final = next((e for e in panel.entries if e.name.startswith("FinalCheck_")), None)
    if final:
        m = final.metrics or {}
        logger.info(f"[Result] Success mode={SUCCESS_MODE} final_flip={m.get('flipped')} final_target={m.get('success_target')} ssim_min met={os.path.exists(os.path.join(desk.master_dir(image_id),'master_meta.json'))}")
    else:
        logger.info("[Result] No final-check entry recorded.")

def orchestrate(image_path: str, fgsm_loops: int = 20, pgd_loops: int = 20, interval: float = 2.0, no_early_stop: bool = False) -> str:
    check_models()
    desk = FileMixingDesk()
    vllm = VLLMClient(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, model=VLLM_MODEL_ID)

    # 1) Save original
    image_id = file_sha1(image_path)
    ConductorAgent(desk).prepare_original(image_path, image_id=image_id)
    logger.info(f"[Conductor] Prepared original | image_id={image_id} | DEVICE_CHOICE={DEVICE_CHOICE}")

    # 1b) Baseline predictions (original image)
    compute_and_save_baseline(desk, image_id)

    # 2) InfoAgent (LLM)
    info = InfoAgent(vllm_client=vllm, mixing_desk=desk, temperature=INFOAGENT_TEMPERATURE, max_tokens=INFOAGENT_MAX_TOKENS)
    info.run(image_source=image_path, image_id=image_id, save_to_mixing_desk=True)

    # 3) ConductorLLM sets the objective from image + InfoAgent output
    obj = ConductorLLMAgent(desk, vllm=vllm).run(image_id)
    # Optionally force a fixed target_class (e.g., 0)
    if FORCE_TARGET_CLASS in (0, 1):
        obj.target_class = FORCE_TARGET_CLASS
        desk.save_objective(obj)
        logger.info(f"[Conductor] FORCE_TARGET_CLASS applied: target_class={FORCE_TARGET_CLASS}")

    # Seed a first strategy
    StrategistAgent(desk, vllm=vllm).run_once(image_id)

    stop_event = Event()

    # 4) Method Agents with per-agent LLM advisors
    fgsm_advisor = MethodAdvisor("FGSM_Agent", desk, vllm=vllm)
    pgd_advisor = MethodAdvisor("PGD_Agent", desk, vllm=vllm)
    fgsm = FGSM_Agent(desk, advisor=fgsm_advisor, llm_every_k=3)
    pgd = PGD_Agent(desk, advisor=pgd_advisor, llm_every_k=3)

    t_fgsm = safe_thread(loop_method, "FGSM", fgsm, image_id, fgsm_loops, interval, stop_event)
    t_pgd = safe_thread(loop_method, "PGD", pgd, image_id, pgd_loops, interval, stop_event)

    # 5) Mixer and Critiques (+ FinalCheck)
    t_mixer = safe_thread(loop_mixer, "Mixer", image_id, interval, stop_event, desk)
    t_c_res = safe_thread(loop_critique_resnet, "Critique-ResNet50", image_id, interval, stop_event, desk)
    t_c_den = safe_thread(loop_critique_densenet, "Critique-DenseNet121", image_id, interval, stop_event, desk)
    t_c_per = safe_thread(loop_critique_perceptual, "Critique-Perceptual", image_id, interval, stop_event, desk)
    t_final = safe_thread(loop_final_check, "FinalCheck", image_id, interval, stop_event, desk)

    # 6) Strategist loop
    t_strat = safe_thread(loop_strategist, "Strategist", image_id, interval, stop_event, desk, vllm)

    t_status = safe_thread(loop_status, "Status", image_id, STATUS_EVERY_SEC, stop_event, desk)

    t_guard = safe_thread(methods_guard, "MethodsGuard", t_fgsm, t_pgd, stop_event)

    # 7) Monitor
    monitor_early_stop(image_id=image_id, desk=desk, stop_event=stop_event, interval=interval, no_early_stop=no_early_stop)

    time.sleep(0.5)
    logger.info("[Main] Joining threads...")
    for t in (t_fgsm, t_pgd, t_mixer, t_c_res, t_c_den, t_c_per, t_final, t_strat, t_status, t_guard):
        t.join(timeout=5.0)

    # Final summary
    master_meta_path = os.path.join(desk.master_dir(image_id), "master_meta.json")
    if os.path.exists(master_meta_path):
        try:
            with open(master_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})
            logger.info(
                f"[Final] avg_conf={float(metrics.get('avg_conf') or 0.0):.4f} "
                f"ssim={float(metrics.get('ssim') or 0.0):.4f} "
                f"master={meta.get('image_path')}"
            )
        except Exception as e:
            logger.warning(f"[Final] Could not read master meta: {e}")
    panel_path = os.path.join(desk.feedback_dir(image_id), "panel.json")
    if os.path.exists(panel_path):
        with open(panel_path, "r", encoding="utf-8") as f:
            panel = json.load(f)
        logger.info(f"[Final] Feedback panel entries: {[e.get('name') for e in panel.get('entries', [])]}")
    logger.info("[Main] Done.")
    return image_id

def loop_status(image_id: str, interval: float, stop_event: Event, desk: FileMixingDesk):
    def safe_get_metric(m: dict, k: str):
        try:
            v = m.get(k, None)
            return f"{float(v):.4f}" if isinstance(v, (int, float)) else "-"
        except Exception:
            return "-"
    while not stop_event.is_set():
        try:
            obj = desk.load_objective(image_id)
            panel = desk.load_feedback_panel(image_id)
            meta = {}
            mpath = os.path.join(desk.master_dir(image_id), "master_meta.json")
            if os.path.exists(mpath):
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            metrics = meta.get("metrics", {}) or {}
            def find(name: str):
                for e in panel.entries:
                    if e.name == name:
                        return e.metrics or {}
                return {}
            res = find("SurrogateCritique_resnet50")
            den = find("SurrogateCritique_densenet121")
            fin = find(f"FinalCheck_{FINALCHECK_NAME}")
            line = (
                f"[Status] Obj(mode={SUCCESS_MODE} tclass={getattr(obj,'target_class',0)} "
                f"ssim_min={getattr(obj,'ssim_min',0.0):.2f} conf_target={getattr(obj,'conf_target',0.0):.2f}) | "
                f"Master(ssim={safe_get_metric(metrics,'ssim')} avg_conf={safe_get_metric(metrics,'avg_conf')}) | "
                f"Res(pred={res.get('pred','-')} flip={res.get('flipped','-')} ct={safe_get_metric(res,'conf_target')}) | "
                f"Den(pred={den.get('pred','-')} flip={den.get('flipped','-')} ct={safe_get_metric(den,'conf_target')}) | "
                f"ViT(pred={fin.get('pred','-')} flip={fin.get('flipped','-')} ct={safe_get_metric(fin,'conf_target')} "
                f"succ_tgt={fin.get('success_target','-')})"
            )
            print(line)
        except Exception as e:
            logger.warning(f"[Status] failed: {e}")
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(0.1)

def methods_guard(t1: Thread, t2: Thread, stop_event: Event):
    # Wait for both method loops to finish
    t1.join()
    t2.join()
    # If objective wasn’t met already, stop the rest
    if not stop_event.is_set():
        logger.info("[Guard] Method loops completed without success; stopping (failure).")
        stop_event.set()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run full LLM-coordinated attacker lineup on a single image (flip-aware).")
    p.add_argument("--image", required=True, help="Path to original image.")
    p.add_argument("--fgsm-loops", type=int, default=20)
    p.add_argument("--pgd-loops", type=int, default=20)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--no-early-stop", action="store_true")
    args = p.parse_args()

    try:
        orchestrate(
            image_path=args.image,
            fgsm_loops=args.fgsm_loops,
            pgd_loops=args.pgd_loops,
            interval=args.interval,
            no_early_stop=args.no_early_stop,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)