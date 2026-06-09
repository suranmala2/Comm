"""
Chronicle Image Gen — Lightning AI Cloud Deploy (Option B: Persistent Volume)
WAI Illustrious SDXL + LoRA support
Endpoints: /txt2img, /img2img, /download_model, /download_lora, /list_loras

HOW TO DEPLOY (scale-to-zero serverless):
  1. pip install litserve
  2. litserve login
  3. litserve deploy lightning_app.py --gpu l4

Lightning AI mounts a persistent volume at /data automatically.
Models written there survive container spin-down — no re-download on cold starts.
First time only: call POST /download_model to populate the volume.
"""

import io
import base64
import os
from pathlib import Path

# ── Storage ───────────────────────────────────────────────────────────────────
# /data is the persistent volume path on Lightning AI cloud deployments.
# Everything here survives scale-to-zero restarts.
VOLUME_PATH = Path(os.environ.get("MODEL_DIR", "/data/models"))
MODEL_PATH  = VOLUME_PATH / "base_model.safetensors"
LORA_DIR    = VOLUME_PATH / "loras"

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


import litserve as ls
from fastapi import Request
from fastapi.responses import JSONResponse


class ChronicleImageGenAPI(ls.LitAPI):

    def setup(self, device: str):
        """
        Runs ONCE per cold start — equivalent to Modal's @modal.enter().
        If the model is already on the persistent volume, load it into GPU now
        so the first real request doesn't pay the pipeline-load cost.
        """
        self.device        = device
        self._pipe         = None
        self._img2img      = None
        self._loaded_loras = []

        if MODEL_PATH.exists():
            self._load_pipeline()
        else:
            print(
                "No model on persistent volume yet. "
                "Call POST /download_model once to populate /data/models."
            )

    def _load_pipeline(self):
        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            DPMSolverMultistepScheduler,
        )
        print(f"Loading pipeline from {MODEL_PATH} ...")
        self._pipe = StableDiffusionXLPipeline.from_single_file(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config, use_karras_sigmas=True
        )
        self._pipe.enable_xformers_memory_efficient_attention()
        self._img2img      = StableDiffusionXLImg2ImgPipeline(**self._pipe.components)
        self._loaded_loras = []
        print("Pipeline ready ✓")

    def _ensure_pipeline(self) -> bool:
        if self._pipe is not None:
            return True
        if MODEL_PATH.exists():
            self._load_pipeline()
        return self._pipe is not None

    def _sync_loras(self, lora_names: list, lora_scale: float):
        if set(self._loaded_loras) != set(lora_names):
            if self._loaded_loras:
                self._pipe.unload_lora_weights()
            for name in lora_names:
                lora_path = LORA_DIR / f"{name}.safetensors"
                if not lora_path.exists():
                    raise FileNotFoundError(f"LoRA '{name}' not found. Call /download_lora first.")
                self._pipe.load_lora_weights(
                    str(LORA_DIR), weight_name=f"{name}.safetensors", adapter_name=name,
                )
            self._loaded_loras = list(lora_names)
        if lora_names:
            self._pipe.set_adapters(lora_names, adapter_weights=[lora_scale] * len(lora_names))

    # LitServe stubs — real logic lives in FastAPI routes below
    def decode_request(self, request): return request
    def predict(self, request):        return request
    def encode_response(self, output): return output

    def _register_routes(self, app):
        import requests as req_lib

        @app.post("/download_model")
        async def download_model(request: Request):
            """
            Download the base model to /data (persistent volume).
            Only needs to be called ONCE ever — survives all future cold starts.
            Body: { "url": "...", "api_key": "optional_civitai_key" }
            """
            body    = await request.json()
            url     = body.get("url", "").strip()
            api_key = body.get("api_key", "").strip()

            if not url:
                return JSONResponse({"error": "url is required"}, status_code=400)

            # If already downloaded, skip — persistent volume still has it
            if MODEL_PATH.exists():
                return JSONResponse({
                    "status": "already_exists",
                    "size_mb": round(MODEL_PATH.stat().st_size / 1e6),
                    "message": "Model is already on the persistent volume.",
                })

            if api_key and "civitai.com" in url and "token=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}token={api_key}"

            VOLUME_PATH.mkdir(parents=True, exist_ok=True)
            print("Downloading base model to persistent volume (/data/models)...")
            with req_lib.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                done = 0
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        done += len(chunk)
            print(f"Downloaded {done / 1e6:.1f} MB")

            size = MODEL_PATH.stat().st_size
            if size < 1_000_000_000:
                MODEL_PATH.unlink()
                return JSONResponse(
                    {"error": "File too small — download likely failed. Check URL and API key."},
                    status_code=400,
                )

            self._ensure_pipeline()   # load into GPU right now
            return JSONResponse({"status": "ok", "size_mb": round(size / 1e6)})

        @app.post("/download_lora")
        async def download_lora(request: Request):
            """
            Download a LoRA to /data (persistent volume). Run once per LoRA.
            Body: { "url": "...", "name": "my_lora", "api_key": "optional_key" }
            """
            body    = await request.json()
            url     = body.get("url", "").strip()
            name    = body.get("name", "").strip()
            api_key = body.get("api_key", "").strip()

            if not url or not name:
                return JSONResponse({"error": "url and name are required"}, status_code=400)

            if api_key and "civitai.com" in url and "token=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}token={api_key}"

            LORA_DIR.mkdir(parents=True, exist_ok=True)
            dest = LORA_DIR / f"{name}.safetensors"
            print(f"Downloading LoRA '{name}' to persistent volume...")
            with req_lib.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            return JSONResponse({"status": "ok", "name": name, "size_mb": round(dest.stat().st_size / 1e6)})

        @app.get("/list_loras")
        async def list_loras():
            LORA_DIR.mkdir(parents=True, exist_ok=True)
            loras = [
                {"name": p.stem, "size_mb": round(p.stat().st_size / 1e6)}
                for p in sorted(LORA_DIR.glob("*.safetensors"))
            ]
            return JSONResponse({
                "model_ready":     MODEL_PATH.exists(),
                "pipeline_loaded": self._pipe is not None,
                "loras":           loras,
            })

        @app.post("/txt2img")
        async def txt2img(request: Request):
            import torch
            body       = await request.json()
            prompt     = body.get("prompt", "")
            neg        = body.get("negative_prompt", "")
            loras      = body.get("loras") or []
            lora_scale = float(body.get("lora_scale", 0.8))
            width      = int(body.get("width",  1024))
            height     = int(body.get("height", 1024))
            steps      = int(body.get("steps",  20))
            cfg        = float(body.get("cfg",   7.0))

            if not prompt:
                return JSONResponse({"error": "prompt is required"}, status_code=400)
            if not self._ensure_pipeline():
                return JSONResponse({"error": "Model not loaded. Call /download_model first."}, status_code=503)

            print(f"[txt2img] {prompt!r} | loras={loras} | {width}x{height} | steps={steps}")
            try:
                self._sync_loras(loras, lora_scale)
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            with torch.inference_mode():
                result = self._pipe(
                    prompt=prompt, negative_prompt=neg,
                    width=width, height=height,
                    num_inference_steps=steps, guidance_scale=cfg,
                ).images[0]
            return JSONResponse({"image": pil_to_b64(result)})

        @app.post("/img2img")
        async def img2img(request: Request):
            import torch
            body       = await request.json()
            prompt     = body.get("prompt", "")
            neg        = body.get("negative_prompt", "")
            img_b64    = body.get("image", "")
            loras      = body.get("loras") or []
            lora_scale = float(body.get("lora_scale", 0.8))
            strength   = float(body.get("strength", 0.55))
            steps      = int(body.get("steps",  20))
            cfg        = float(body.get("cfg",   7.0))

            if not prompt or not img_b64:
                return JSONResponse({"error": "prompt and image are required"}, status_code=400)
            if not self._ensure_pipeline():
                return JSONResponse({"error": "Model not loaded. Call /download_model first."}, status_code=503)

            print(f"[img2img] {prompt!r} | loras={loras} | strength={strength} | steps={steps}")
            try:
                self._sync_loras(loras, lora_scale)
            except FileNotFoundError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            source = b64_to_pil(img_b64)
            with torch.inference_mode():
                result = self._img2img(
                    prompt=prompt, negative_prompt=neg, image=source,
                    strength=strength, num_inference_steps=steps, guidance_scale=cfg,
                ).images[0]
            return JSONResponse({"image": pil_to_b64(result)})


if __name__ == "__main__":
    api    = ChronicleImageGenAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",      # auto-detects GPU; works locally (cpu) and on cloud (gpu)
        devices=1,
        workers_per_device=1,    # 1 worker = model stays in VRAM between requests
        timeout=300,
    )
    api._register_routes(server.app)
    server.run(port=int(os.environ.get("PORT", 8000)))
