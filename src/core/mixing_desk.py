import os
import json
import tempfile
from datetime import datetime
from filelock import FileLock
from .schemas import InfoAgentReport, ConductorObjective, TrackMeta, MasterMeta, CritiqueEntry, FeedbackPanel, StrategyDoc

def _atomic_write_json(path: str, data: dict):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)

class FileMixingDesk:
    def __init__(self, base_dir: str = "./data/mixing_desk"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _image_dir(self, image_id: str) -> str:
        d = os.path.join(self.base_dir, image_id)
        os.makedirs(d, exist_ok=True)
        return d

    def path_original(self, image_id: str) -> str:
        d = self._image_dir(image_id)
        p = os.path.join(d, "original", "image.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def path_objective(self, image_id: str) -> str:
        d = self._image_dir(image_id)
        return os.path.join(d, "conductor_objective.json")

    def tracks_dir(self, image_id: str) -> str:
        return os.path.join(self._image_dir(image_id), "tracks")

    def track_dir(self, image_id: str, agent_name: str) -> str:
        d = os.path.join(self.tracks_dir(image_id), agent_name)
        os.makedirs(d, exist_ok=True)
        return d

    def master_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "master")
        os.makedirs(d, exist_ok=True)
        return d

    def feedback_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "feedback")
        os.makedirs(d, exist_ok=True)
        return d

    def strategy_dir(self, image_id: str) -> str:
        d = os.path.join(self._image_dir(image_id), "strategy")
        os.makedirs(d, exist_ok=True)
        return d

    # Info report
    def save_info_report(self, report: InfoAgentReport) -> str:
        d = self._image_dir(report.image_id)
        fname = f"info_report_{report.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(d, fname)
        _atomic_write_json(fpath, report.model_dump())
        _atomic_write_json(os.path.join(d, "info_report_latest.json"), report.model_dump())
        return fpath

    # Objective
    def save_objective(self, obj: ConductorObjective):
        path = self.path_objective(obj.image_id)
        _atomic_write_json(path, obj.model_dump())
        return path

    def load_objective(self, image_id: str) -> ConductorObjective:
        path = self.path_objective(image_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ConductorObjective(**data)

    # Track IO
    def write_track(self, image_id: str, agent_name: str, meta: TrackMeta):
        d = self.track_dir(image_id, agent_name)
        latest_meta = os.path.join(d, "latest_meta.json")
        _atomic_write_json(latest_meta, meta.model_dump())
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hist_meta = os.path.join(d, f"meta_{ts}_step{meta.step}.json")
        _atomic_write_json(hist_meta, meta.model_dump())

    def list_tracks(self, image_id: str):
        d = self.tracks_dir(image_id)
        if not os.path.exists(d):
            return []
        return [name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name))]

    # Master IO
    def write_master(self, image_id: str, meta: MasterMeta):
        d = self.master_dir(image_id)
        latest_meta = os.path.join(d, "master_meta.json")
        _atomic_write_json(latest_meta, meta.model_dump())

    # Feedback (locked + atomic)
    def append_feedback(self, image_id: str, entry: CritiqueEntry):
        d = self.feedback_dir(image_id)
        panel_path = os.path.join(d, "panel.json")
        lock = FileLock(panel_path + ".lock")
        with lock:
            if os.path.exists(panel_path):
                try:
                    with open(panel_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    panel = FeedbackPanel(**data)
                except Exception:
                    panel = FeedbackPanel(image_id=image_id)  # recover from corruption
            else:
                panel = FeedbackPanel(image_id=image_id)
            panel.entries = [e for e in panel.entries if e.name != entry.name]
            panel.entries.append(entry)
            panel.updated_at = datetime.utcnow()
            _atomic_write_json(panel_path, panel.model_dump())
            _atomic_write_json(panel_path + ".bak", panel.model_dump())

    def load_feedback_panel(self, image_id: str) -> FeedbackPanel:
        d = self.feedback_dir(image_id)
        panel_path = os.path.join(d, "panel.json")
        lock = FileLock(panel_path + ".lock")
        with lock.acquire(timeout=1.0):
            if not os.path.exists(panel_path):
                return FeedbackPanel(image_id=image_id)
            try:
                with open(panel_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return FeedbackPanel(**data)
            except Exception:
                # Try backup
                bak = panel_path + ".bak"
                if os.path.exists(bak):
                    try:
                        with open(bak, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        return FeedbackPanel(**data)
                    except Exception:
                        pass
                return FeedbackPanel(image_id=image_id)

    # Strategy IO
    def path_strategy(self, image_id: str) -> str:
        return os.path.join(self.strategy_dir(image_id), "strategy.json")

    def save_strategy(self, strategy: StrategyDoc):
        p = self.path_strategy(strategy.image_id)
        _atomic_write_json(p, strategy.model_dump())
        return p

    def load_strategy(self, image_id: str) -> StrategyDoc | None:
        p = self.path_strategy(image_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StrategyDoc(**data)

    # Baseline store/load (unchanged from your version)
    def path_baseline(self, image_id: str) -> str:
        return os.path.join(self._image_dir(image_id), "baseline.json")

    def save_baseline(self, image_id: str, data: dict):
        p = self.path_baseline(image_id)
        _atomic_write_json(p, data)
        return p

    def load_baseline(self, image_id: str) -> dict | None:
        p = self.path_baseline(image_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)