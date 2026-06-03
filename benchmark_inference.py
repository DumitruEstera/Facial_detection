#!/usr/bin/env python3
"""
benchmark_inference.py — Măsurarea latenței de inferență per detector.

Rulează fiecare modul AI din pipeline (față, plăcuțe, foc, armă, HAR) pe un set
de cadre reprezentative și raportează latența reală (mediană / medie / p95) pe
cadru. Cifrele rezultate pot înlocui estimările orientative din figura
pipeline-ului (§4.4) cu valori măsurate.

IMPORTANT — ce măsoară și ce NU măsoară:
  * Măsoară DOAR timpul de inferență (forward pass + pre/post-procesarea din
    `process_frame`). Inițializarea modelelor (încărcare pe GPU) NU este numărată.
  * NU trebuie să declanșezi alerte reale (bătăi, foc, arme). Latența unui model
    este aproape independentă de faptul că *găsește* sau nu ceva — depinde de
    rezoluție, arhitectură și numărul de obiecte detectate.
  * Pentru cifre realiste la FAȚĂ și PLĂCUȚE folosește un video care chiar
    conține o față / o plăcuță (costul scalează per obiect: recunoaștere FAISS
    per față, EasyOCR per plăcuță). Pentru FOC / ARMĂ / HAR conținutul nu
    contează semnificativ — orice video merge.

Exemple:
    # video reprezentativ, toate modulele
    python benchmark_inference.py --source /cale/clip.mp4 --frames 200

    # doar modulele care nu au nevoie de PostgreSQL
    python benchmark_inference.py --source /cale/clip.mp4 --modules fire weapon har

    # webcam
    python benchmark_inference.py --source 0

    # fără sursă reală (sintetic) — doar pentru smoke-test; cifrele față/plăcuțe
    # vor fi subestimate (0 obiecte detectate)
    python benchmark_inference.py --synthetic
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

# Rulează din directorul scriptului ca să se rezolve căile implicite ale
# modelelor ("models/...") exact ca în app.py.
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Aceleași valori implicite ca în app.py (suprascriabile din environment).
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "facial_recognition"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "incorect"),
}


def cuda_sync():
    """Forțează finalizarea kernel-urilor GPU înainte de a opri cronometrul.

    Lansările CUDA sunt asincrone; fără sync am măsura timpul de *lansare* a
    kernel-ului, nu timpul de *execuție*."""
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
#                              Sursa de cadre                                 #
# --------------------------------------------------------------------------- #
class FrameSource:
    """Iterator de cadre dintr-un video / index webcam / director de imagini.

    Buclează la nesfârșit (reia de la capăt) ca să poată servi oricâte cadre
    cere benchmark-ul."""

    def __init__(self, source: str, width: int, height: int):
        self.size = (width, height)
        self._images: Optional[List[Path]] = None
        self._img_idx = 0
        self.cap = None

        p = Path(source)
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            self.kind = f"webcam[{source}]"
        elif p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._images = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if not self._images:
                raise RuntimeError(f"Niciun fișier imagine în directorul {p}")
            self.kind = f"imagini[{len(self._images)}]"
        elif p.is_file():
            self.cap = cv2.VideoCapture(str(p))
            self.kind = f"video[{p.name}]"
        else:
            raise RuntimeError(f"Sursă invalidă: {source}")

        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"Nu pot deschide sursa video: {source}")

    def read(self) -> np.ndarray:
        if self._images is not None:
            frame = cv2.imread(str(self._images[self._img_idx % len(self._images)]))
            self._img_idx += 1
        else:
            ok, frame = self.cap.read()
            if not ok:  # video epuizat → reia de la început
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Nu pot citi cadre din sursă")
        if self.size[0] > 0 and self.size[1] > 0:
            frame = cv2.resize(frame, self.size)
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class SyntheticSource:
    """Cadre de zgomot — doar pentru smoke-test (nu detectează nimic real)."""

    def __init__(self, width: int, height: int):
        self.size = (width or 640, height or 480)
        self.kind = "sintetic[zgomot]"

    def read(self) -> np.ndarray:
        return np.random.randint(0, 256, (self.size[1], self.size[0], 3), dtype=np.uint8)

    def release(self):
        pass


# --------------------------------------------------------------------------- #
#                          Inițializarea detectoarelor                        #
# --------------------------------------------------------------------------- #
def make_face():
    from facial_recognition_system import FacialRecognitionSystem
    sys_ = FacialRecognitionSystem(DB_CONFIG, camera_id="0")
    # process_frame -> (frame_adnotat, rezultate); măsurăm tot apelul.
    return lambda frame: sys_.process_frame(frame)


def make_plate():
    from license_plate_recognition_system import LicensePlateRecognitionSystem
    sys_ = LicensePlateRecognitionSystem(DB_CONFIG)
    return lambda frame: sys_.process_frame(frame)


def make_fire():
    from fire_detection_system import FireDetectionSystem
    sys_ = FireDetectionSystem()
    return lambda frame: sys_.process_frame(frame)


def make_weapon():
    from weapon_detection_system import WeaponDetectionSystem
    sys_ = WeaponDetectionSystem(model_path="models/weapon/best.pt")
    return lambda frame: sys_.process_frame(frame)


def make_har():
    from har_system import HumanActionRecognitionSystem
    # clip_interval_frames=1 => rulează inferența SlowFast la FIECARE cadru,
    # cu condiția să avem ≥32 cadre în buffer (le pre-umplem la warmup).
    sys_ = HumanActionRecognitionSystem(
        model_path="models/har/best_model.pth",
        device="auto",
        confidence_threshold=0.5,
        clip_interval_frames=1,
    )
    return sys_  # întoarcem obiectul: HAR are nevoie de tratare specială


MODULE_FACTORIES: dict = {
    "face": make_face,
    "plate": make_plate,
    "fire": make_fire,
    "weapon": make_weapon,
    "har": make_har,
}
NEEDS_DB = {"face", "plate"}


# --------------------------------------------------------------------------- #
#                              Bucla de măsurare                               #
# --------------------------------------------------------------------------- #
def bench_callable(name: str, fn: Callable, src, frames: int, warmup: int) -> List[float]:
    """Cronometrează `fn(frame)` pe `frames` cadre, după `warmup` rulări de
    încălzire (prima inferență pe GPU e mereu mult mai lentă)."""
    for _ in range(warmup):
        fn(src.read())
    cuda_sync()

    times_ms: List[float] = []
    for _ in range(frames):
        frame = src.read()
        t0 = time.perf_counter()
        fn(frame)
        cuda_sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def bench_har(name: str, har, src, frames: int, warmup: int) -> List[float]:
    """HAR e special: `process_frame` doar acumulează în ring buffer și rulează
    SlowFast periodic. Pre-umplem bufferul (≥64 cadre) la warmup ca fiecare apel
    măsurat să declanșeze o inferență completă, apoi citim `last_inference_time`
    raportat chiar de modul (timpul pur al forward-pass-ului)."""
    for _ in range(max(warmup, 64)):
        har.process_frame(src.read())
    cuda_sync()

    times_ms: List[float] = []
    for _ in range(frames):
        har.process_frame(src.read())
        cuda_sync()
        # last_inference_time e setat în har_system.py în jurul forward-pass-ului
        times_ms.append(har.last_inference_time * 1000.0)
    return times_ms


def summarize(times_ms: List[float]) -> dict:
    s = sorted(times_ms)
    n = len(s)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n": n,
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "p95": p95,
        "min": s[0],
        "max": s[-1],
        "fps": 1000.0 / statistics.median(s) if s and s[len(s) // 2] > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
#                                   main                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Benchmark latență inferență per detector")
    ap.add_argument("--source", default=None,
                    help="video, director de imagini, sau index webcam (ex. 0)")
    ap.add_argument("--synthetic", action="store_true",
                    help="cadre de zgomot în loc de o sursă reală (doar smoke-test)")
    ap.add_argument("--modules", nargs="+", default=list(MODULE_FACTORIES),
                    choices=list(MODULE_FACTORIES),
                    help="ce module să măsoare (implicit: toate)")
    ap.add_argument("--frames", type=int, default=200, help="cadre măsurate / modul")
    ap.add_argument("--warmup", type=int, default=10, help="cadre de încălzire / modul")
    ap.add_argument("--width", type=int, default=640, help="lățime resize (0 = nativ)")
    ap.add_argument("--height", type=int, default=480, help="înălțime resize (0 = nativ)")
    args = ap.parse_args()

    if not args.synthetic and not args.source:
        ap.error("dă fie --source <cale/index>, fie --synthetic")

    # Informații despre mediu (esențiale pentru a cita cifrele în lucrare).
    print("=" * 70)
    if _HAS_TORCH and torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"CUDA     : {torch.version.cuda}  |  torch {torch.__version__}")
    else:
        print("GPU      : indisponibil — rulez pe CPU (cifrele NU sunt reprezentative)")
    print(f"Rezoluție: {args.width}x{args.height}")
    print(f"Cadre    : {args.frames} măsurate, {args.warmup} warmup, per modul")

    results: dict = {}
    for name in args.modules:
        print("\n" + "-" * 70)
        print(f"▶ Modul: {name}")
        if name in NEEDS_DB:
            print(f"  (necesită PostgreSQL @ {DB_CONFIG['host']}/{DB_CONFIG['database']})")
        try:
            obj = MODULE_FACTORIES[name]()
        except Exception as e:
            print(f"  ⚠️  SĂRIT — inițializare eșuată: {type(e).__name__}: {e}")
            continue

        # O sursă proaspătă per modul (de la primul cadru).
        try:
            src = (SyntheticSource(args.width, args.height) if args.synthetic
                   else FrameSource(args.source, args.width, args.height))
        except Exception as e:
            print(f"  ⚠️  SĂRIT — sursă invalidă: {e}")
            continue
        print(f"  Sursă: {src.kind}")

        try:
            if name == "har":
                times = bench_har(name, obj, src, args.frames, args.warmup)
            else:
                times = bench_callable(name, obj, src, args.frames, args.warmup)
        except Exception as e:
            print(f"  ⚠️  EROARE la măsurare: {type(e).__name__}: {e}")
            src.release()
            continue
        src.release()

        results[name] = summarize(times)
        r = results[name]
        print(f"  mediană={r['median']:.1f} ms  medie={r['mean']:.1f} ms  "
              f"p95={r['p95']:.1f} ms  min={r['min']:.1f}  max={r['max']:.1f}  "
              f"(~{r['fps']:.0f} fps)")

    if not results:
        print("\nNiciun modul măsurat cu succes.")
        return

    # ── Tabel rezumat ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REZUMAT (latență de inferență per cadru/clip)")
    print("=" * 70)
    hdr = f"{'modul':<10}{'mediană':>10}{'medie':>10}{'p95':>10}{'min':>9}{'max':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<10}{r['median']:>9.1f}{r['mean']:>10.1f}{r['p95']:>10.1f}"
              f"{r['min']:>9.1f}{r['max']:>9.1f}")

    # ── Tabel LaTeX gata de lipit în §4.4 ─────────────────────────────────
    print("\n" + "=" * 70)
    print("Tabel LaTeX (booktabs):")
    print("=" * 70)
    label = {"face": "Recunoaștere facială", "plate": "Plăcuțe înmatriculare",
             "fire": "Detecție foc/fum", "weapon": "Detecție armă",
             "har": "Recunoaștere acțiuni (SlowFast)"}
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Detector & Mediană (ms) & p95 (ms) & Throughput (fps) \\")
    print(r"\midrule")
    for name, r in results.items():
        print(f"{label.get(name, name)} & {r['median']:.1f} & {r['p95']:.1f} & {r['fps']:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    sys.exit(main())
