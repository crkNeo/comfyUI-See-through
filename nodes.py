import os
import sys
import random
import uuid
from datetime import datetime

print("[SeeThrough] nodes.py: starting imports...", flush=True)

import torch
import numpy as np

import folder_paths
import comfy.model_management as mm
import comfy.utils


def _log_vram(label):
    """Log current GPU VRAM usage for profiling."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[SeeThrough VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)


def _pick_dtype(use_nf4=False):
    """bf16 on Ampere+ (sm_80+), fp16 on older cards.

    Turing (sm_75, e.g. GTX 16xx / RTX 20xx) has no native bf16 — torch emulates it,
    which is correct but slow. fp16 runs on the real tensor cores there instead.

    NF4 is the exception: the published NF4 checkpoints carry
    bnb_4bit_compute_dtype=bfloat16, so the 4-bit Linear layers dequantize to bf16.
    Casting the surrounding non-quantized tensors to fp16 mixes dtypes inside
    attention and crashes with "CUDA error: an illegal memory access was
    encountered". In NF4 mode the checkpoint wins, whatever the card prefers.

    Override with SEETHROUGH_DTYPE=bf16|fp16|fp32 if a card misbehaves.
    """
    override = os.environ.get("SEETHROUGH_DTYPE", "").strip().lower()
    forced = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
              "fp16": torch.float16, "float16": torch.float16,
              "fp32": torch.float32, "float32": torch.float32}.get(override)
    if forced is not None:
        dtype, why = forced, f"SEETHROUGH_DTYPE={override}"
    elif use_nf4:
        dtype, why = torch.bfloat16, "nf4 checkpoint declares bnb_4bit_compute_dtype=bfloat16"
    elif not torch.cuda.is_available():
        dtype, why = torch.float32, "no CUDA device"
    else:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            dtype, why = torch.bfloat16, f"sm_{major}{minor} has native bf16"
        else:
            dtype, why = torch.float16, f"sm_{major}{minor} lacks native bf16, using fp16"

    print(f"[SeeThrough] dtype = {dtype} ({why})", flush=True)
    return dtype


print("[SeeThrough] nodes.py: comfy imports OK", flush=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEETHROUGH_ROOT_DIR = os.path.join(CURRENT_DIR, "see-through")
SEETHROUGH_COMMON_DIR = os.path.join(SEETHROUGH_ROOT_DIR, "common")

print(f"[SeeThrough] CURRENT_DIR = {CURRENT_DIR}", flush=True)
print(f"[SeeThrough] SEETHROUGH_COMMON_DIR = {SEETHROUGH_COMMON_DIR}", flush=True)
print(f"[SeeThrough] common dir exists = {os.path.isdir(SEETHROUGH_COMMON_DIR)}", flush=True)

# Mock pycocotools if not installed (only used for mask RLE, not needed here)
try:
    import pycocotools  # noqa: F401
    print("[SeeThrough] pycocotools found", flush=True)
except ImportError:
    print("[SeeThrough] pycocotools not found, installing mock...", flush=True)
    import types as _types
    _mock_pycocotools = _types.ModuleType("pycocotools")
    _mock_mask = _types.ModuleType("pycocotools.mask")
    _mock_pycocotools.mask = _mock_mask
    sys.modules["pycocotools"] = _mock_pycocotools
    sys.modules["pycocotools.mask"] = _mock_mask

if SEETHROUGH_COMMON_DIR not in sys.path:
    sys.path.insert(0, SEETHROUGH_COMMON_DIR)
    print(f"[SeeThrough] Added to sys.path: {SEETHROUGH_COMMON_DIR}", flush=True)

if SEETHROUGH_ROOT_DIR not in sys.path:
    sys.path.insert(1, SEETHROUGH_ROOT_DIR)
    print(f"[SeeThrough] Added to sys.path: {SEETHROUGH_ROOT_DIR}", flush=True)

_st_conflict_backup = {}
for _prefix in ("utils", "modules"):
    for _key in list(sys.modules.keys()):
        if _key == _prefix or _key.startswith(_prefix + "."):
            _st_conflict_backup[_key] = sys.modules.pop(_key)
if _st_conflict_backup:
    print(f"[SeeThrough] Temporarily removed {len(_st_conflict_backup)} conflicting sys.modules entries: "
          f"{list(_st_conflict_backup.keys())[:10]}{'...' if len(_st_conflict_backup) > 10 else ''}", flush=True)

print("[SeeThrough] Importing see-through modules...", flush=True)
import cv2
from safetensors.torch import load_file

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, img_alpha_blending, smart_resize
from utils.torchcv import cluster_inpaint_part

print("[SeeThrough] All see-through imports OK", flush=True)

from .split_utils import (compute_labels, split_part_by_labels, label_overlay,
                          unload_lama, SPLIT_MODES)

for _key, _mod in _st_conflict_backup.items():
    if _key not in sys.modules:
        sys.modules[_key] = _mod
del _st_conflict_backup

DEFAULT_LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
DEFAULT_LAYERDIFF_NF4_REPO = "24yearsold/seethroughv0.0.2_layerdiff3d_nf4"
DEFAULT_DEPTH_REPO = "layerdifforg/seethroughv0.0.1_marigold"
DEFAULT_DEPTH_NF4_REPO = "24yearsold/seethroughv0.0.1_marigold_nf4"

QUANT_MODES = ["none", "nf4"]

_NF4_REPO_MAP = {
    DEFAULT_LAYERDIFF_REPO: DEFAULT_LAYERDIFF_NF4_REPO,
    DEFAULT_DEPTH_REPO: DEFAULT_DEPTH_NF4_REPO,
}

VALID_BODY_PARTS_V2 = [
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
    "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
]

folder_paths.add_model_folder_path("SeeThrough", os.path.join(folder_paths.models_dir, "SeeThrough"))


def _model_base_dirs():
    """All registered SeeThrough model dirs, extra_model_paths.yaml entries first when is_default."""
    return folder_paths.get_folder_paths("SeeThrough")


try:
    os.makedirs(_model_base_dirs()[0], exist_ok=True)
except OSError:
    pass


class SeeThrough_LayersData:
    """Output of GenerateLayers: raw RGBA layers + preprocessing info."""
    def __init__(self, layer_dict, fullpage, input_img, resolution, pad_size, pad_pos):
        self.layer_dict = layer_dict      # tag -> RGBA numpy (resolution x resolution)
        self.fullpage = fullpage           # center-padded input (resolution x resolution, RGBA)
        self.input_img = input_img         # original input (RGBA)
        self.resolution = resolution
        self.pad_size = pad_size
        self.pad_pos = pad_pos
        self.scale = pad_size[0] / resolution


class SeeThrough_LayersDepthData:
    """Output of GenerateDepth: layers + per-tag depth maps."""
    def __init__(self, layer_dict, depth_dict, fullpage, resolution):
        self.layer_dict = layer_dict      # tag -> RGBA numpy
        self.depth_dict = depth_dict      # tag -> float32 depth [0,1]
        self.fullpage = fullpage
        self.resolution = resolution


def _cast_non_quantized_params(model, dtype):
    """Cast non-quantized float parameters to dtype, leaving bitsandbytes 4-bit params and integer tensors untouched."""
    try:
        import bitsandbytes as bnb
        bnb_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        bnb_types = ()
    for module in model.modules():
        if bnb_types and isinstance(module, bnb_types):
            continue
        for param in module.parameters(recurse=False):
            if param.data.is_floating_point():
                param.data = param.data.to(dtype=dtype)
        for buf in module.buffers(recurse=False):
            if buf.data.is_floating_point():
                buf.data = buf.data.to(dtype=dtype)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scan_model_dirs():
    """Recursively find diffusers model dirs (containing model_index.json) up to 2 levels deep
    across every registered SeeThrough folder.
    Returns relative paths from their base dir (e.g. 'foo' or 'org/foo'), deduplicated."""
    found = []
    for base in _model_base_dirs():
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, "model_index.json")) and "." not in found:
            found.append(".")
        for name in sorted(os.listdir(base)):
            d1 = os.path.join(base, name)
            if not os.path.isdir(d1):
                continue
            if os.path.isfile(os.path.join(d1, "model_index.json")):
                if name not in found:
                    found.append(name)
                continue
            for sub in sorted(os.listdir(d1)):
                d2 = os.path.join(d1, sub)
                rel = f"{name}/{sub}"
                if os.path.isdir(d2) and os.path.isfile(os.path.join(d2, "model_index.json")) and rel not in found:
                    found.append(rel)
    return found


def _resolve_model_path(model_name):
    bases = _model_base_dirs()
    if model_name == ".":
        for base in bases:
            if os.path.isfile(os.path.join(base, "model_index.json")):
                return base
        return bases[0]
    for base in bases:
        local = os.path.join(base, model_name)
        if os.path.isdir(local):
            return local
    # When model_name is an org/repo style ID (e.g. "layerdifforg/seethroughv0.0.1_marigold"),
    # check if just the repo part exists locally (git clone creates dirs without the org prefix).
    basename = model_name.split("/")[-1]
    if basename != model_name:
        for base in bases:
            local_basename = os.path.join(base, basename)
            if os.path.isdir(local_basename):
                return local_basename
    return model_name


def _label_lr_split(labels, stats, id1, id2):
    label1 = (labels == id1).astype(np.uint8) * 255
    label2 = (labels == id2).astype(np.uint8) * 255
    stats1, stats2 = stats[id1], stats[id2]
    x1 = stats[id1][0] + stats[id1][2] / 2
    x2 = stats[id2][0] + stats[id2][2] / 2
    if x2 < x1:
        return label2, label1, stats2, stats1
    return label1, label2, stats1, stats2


def _process_cuts(img, depth, src_xyxy, tgt_bbox, mask=None):
    tx1, ty1, tx2, ty2 = tgt_bbox[:4]
    tx2 += tx1
    ty2 += ty1
    img = img[ty1:ty2, tx1:tx2].copy()
    depth = depth[ty1:ty2, tx1:tx2]
    depth_median = 1.0
    if mask is not None:
        mask = (mask[ty1:ty2, tx1:tx2].copy() > 15).astype(np.uint8)
        ksize = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
        mask = cv2.dilate(mask, element)
        img[..., -1] *= mask
        depth = 1 - (1 - depth) * mask
        if np.any(mask):
            depth_median = float(np.median(depth[mask > 0]))
    fxyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1], tx2 + src_xyxy[0], ty2 + src_xyxy[1]]
    return img, depth, fxyxy, depth_median


def _part_lr_split(tag, part_info):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        part_info["mask"].astype(np.uint8) * 255, connectivity=8)
    tag2pinfo = {}
    if len(stats) > 2:
        stats = np.array(stats)
        stats_order = np.argsort(stats[..., -1])[::-1][1:]
        arml_mask, armr_mask, statsl, statsr = _label_lr_split(labels, stats, stats_order[0], stats_order[1])
        img, depth, xyxy, dm = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl, mask=arml_mask)
        tag2pinfo[f"{tag}-r"] = {"img": img, "xyxy": xyxy, "depth": depth, "depth_median": dm, "tag": f"{tag}-r"}
        img, depth, xyxy, dm = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr, mask=armr_mask)
        tag2pinfo[f"{tag}-l"] = {"img": img, "xyxy": xyxy, "depth": depth, "depth_median": dm, "tag": f"{tag}-l"}
    else:
        tag2pinfo[tag] = part_info
    return tag2pinfo


def _tag_lr_split(tag, tag2pinfo):
    if tag in tag2pinfo:
        tag2pinfo.update(_part_lr_split(tag, tag2pinfo.pop(tag)))


def _compute_depth_median(part_dict):
    img = part_dict.pop("img")
    part_dict.pop("mask", None)
    depth = part_dict.pop("depth")
    mask = img[..., -1] > 10
    depth_median = float(np.median(depth[mask])) if np.any(mask) else 1.0
    nz = cv2.findNonZero(mask.astype(np.uint8))
    if nz is not None:
        xywh = cv2.boundingRect(nz)
        cx1, cy1 = int(xywh[0]), int(xywh[1])
        cx2, cy2 = cx1 + int(xywh[2]), cy1 + int(xywh[3])
        depth = depth[cy1:cy2, cx1:cx2]
        img = img[cy1:cy2, cx1:cx2]
        if "xyxy" in part_dict:
            ox, oy = part_dict["xyxy"][0], part_dict["xyxy"][1]
            part_dict["xyxy"] = [ox + cx1, oy + cy1, ox + cx2, oy + cy2]
        else:
            part_dict["xyxy"] = [cx1, cy1, cx2, cy2]
    depth = np.clip(depth, 0, 1) * 255
    depth = np.round(depth).astype(np.uint8)
    part_dict["depth_median"] = depth_median
    part_dict["img"] = img
    part_dict["depth"] = depth
    return part_dict


def _crop_head(img, xywh):
    x, y, w, h = xywh
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h
    if w < iw // 2:
        px = min(iw - x - w, x, w // 5)
        x1 = min(max(x - px, 0), iw)
        x2 = min(max(x + w + px, 0), iw)
    if h < ih // 2:
        py = min(ih - y - h, y, h // 5)
        y2 = min(max(y + h + py, 0), ih)
        y1 = min(max(y - py, 0), ih)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def _make_preview(tag2pinfo, resolution):
    drawables = list(tag2pinfo.values())
    if drawables:
        blended = img_alpha_blending(drawables, premultiplied=False, final_size=(resolution, resolution))
    else:
        blended = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    preview = blended[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(preview).unsqueeze(0)


class SeeThrough_LoadLayerDiffModel:
    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_LAYERDIFF_REPO, DEFAULT_LAYERDIFF_NF4_REPO]
        return {
            "required": {
                "model": (model_list, {"default": DEFAULT_LAYERDIFF_REPO,
                                       "tooltip": "HuggingFace repo ID or local model folder in models/SeeThrough/"}),
            },
            "optional": {
                "vae_ckpt": ("STRING", {"default": "",
                                        "tooltip": "Optional path to a custom VAE checkpoint (.safetensors)"}),
                "unet_ckpt": ("STRING", {"default": "",
                                         "tooltip": "Optional path to a custom UNet checkpoint"}),
                "quant_mode": (QUANT_MODES, {"default": "none",
                                              "tooltip": "Quantization mode: 'none' for bf16, 'nf4' for 4-bit NormalFloat quantization (~8GB VRAM). Requires bitsandbytes."}),
                "cache_tag_embeds": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Pre-compute and cache tag embeddings, then unload text encoders to save VRAM"}),
                "group_offload": ("BOOLEAN", {"default": False,
                                               "tooltip": "Enable group offload to reduce peak VRAM (~10GB) at cost of ~1.5x slower speed"}),
                "auto_download": ("BOOLEAN", {"default": True,
                                               "tooltip": "If model is not found locally, download from HuggingFace. Disable to force local-only and error out instead of downloading."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERDIFF_MODEL",)
    RETURN_NAMES = ("layerdiff_model",)
    FUNCTION = "load_model"
    CATEGORY = "SeeThrough"

    def load_model(self, model, vae_ckpt="", unet_ckpt="", quant_mode="none", cache_tag_embeds=True, group_offload=False, auto_download=True):
        use_nf4 = quant_mode == "nf4"
        dtype = _pick_dtype(use_nf4)

        if use_nf4 and model in _NF4_REPO_MAP:
            model = _NF4_REPO_MAP[model]
            print(f"[SeeThrough] quant_mode=nf4: auto-switched to NF4 repo {model}", flush=True)

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)
        local_only = is_local or not auto_download

        print(f"[SeeThrough] Loading LayerDiff model from: {pretrained} "
              f"(quant_mode={quant_mode}, local={is_local}, local_files_only={local_only})", flush=True)
        trans_vae = TransparentVAE.from_pretrained(pretrained, subfolder="trans_vae", local_files_only=local_only)

        if unet_ckpt:
            print(f"[SeeThrough] Loading custom UNet from: {unet_ckpt}", flush=True)
            unet = UNetFrameConditionModel.from_pretrained(unet_ckpt)
        else:
            unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", local_files_only=local_only)

        pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            pretrained, trans_vae=trans_vae, unet=unet, scheduler=None, local_files_only=local_only)

        if vae_ckpt:
            print(f"[SeeThrough] Loading custom VAE from: {vae_ckpt}", flush=True)
            td_sd, vae_sd = {}, {}
            sd = load_file(vae_ckpt)
            for k, v in sd.items():
                if k.startswith("trans_decoder."):
                    td_sd[k[len("trans_decoder."):]] = v
                elif k.startswith("vae."):
                    vae_sd[k.replace("vae.", "")] = v
            if vae_sd:
                pipeline.vae.load_state_dict(vae_sd)
            if td_sd:
                pipeline.trans_vae.decoder.load_state_dict(td_sd)

        pipeline.vae.to(dtype=dtype)
        pipeline.trans_vae.to(dtype=dtype)

        if use_nf4:
            print(f"[SeeThrough] NF4 mode: casting non-quantized parameters to {dtype}", flush=True)
            _cast_non_quantized_params(pipeline.unet, dtype)
            _cast_non_quantized_params(pipeline.text_encoder, dtype)
            _cast_non_quantized_params(pipeline.text_encoder_2, dtype)
        else:
            pipeline.unet.to(dtype=dtype)
            pipeline.text_encoder.to(dtype=dtype)
            pipeline.text_encoder_2.to(dtype=dtype)

        pipeline._st_group_offload = False
        if group_offload and use_nf4:
            # Group offload moves module blocks on and off the GPU with hooks, but a
            # bitsandbytes Params4bit carries a quant_state holding device pointers that
            # the hook's .to() does not migrate. The next kernel then dereferences a stale
            # pointer and the process dies with "CUDA error: an illegal memory access was
            # encountered" — an abort, not a catchable OOM. NF4 already cuts VRAM enough
            # that offloading on top of it buys little, so quantization wins.
            print("[SeeThrough] WARNING: group_offload is incompatible with quant_mode=nf4 "
                  "(bitsandbytes quant_state is not migrated by the offload hooks, which "
                  "crashes with an illegal memory access). Disabling group_offload; "
                  "NF4 alone is the lower-VRAM option.", flush=True)
            group_offload = False

        if group_offload:
            if hasattr(pipeline, 'enable_group_offload'):
                print("[SeeThrough] Enabling group offload for LayerDiff pipeline", flush=True)
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
                pipeline._st_group_offload = True
            else:
                print("[SeeThrough] WARNING: group_offload requires diffusers >= 0.37.0, skipping. Please upgrade: pip install diffusers>=0.37.0", flush=True)

        if cache_tag_embeds:
            if not pipeline._st_group_offload:
                device = mm.get_torch_device()
                if not use_nf4:
                    pipeline.text_encoder.to(device)
                    pipeline.text_encoder_2.to(device)
            print("[SeeThrough] Caching tag embeddings and unloading text encoders...", flush=True)
            pipeline.cache_tag_embeds(unload_textencoders=True)
            _log_vram("After cache_tag_embeds (text encoders unloaded)")

        _log_vram("LayerDiff model loaded (CPU)")
        print(f"[SeeThrough] LayerDiff model loaded (quant_mode={quant_mode})", flush=True)
        return (pipeline,)

class SeeThrough_LoadDepthModel:
    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_DEPTH_REPO, DEFAULT_DEPTH_NF4_REPO]
        return {
            "required": {
                "model": (model_list, {"default": DEFAULT_DEPTH_REPO,
                                       "tooltip": "HuggingFace repo ID or local model folder in models/SeeThrough/"}),
            },
            "optional": {
                "quant_mode": (QUANT_MODES, {"default": "none",
                                              "tooltip": "Quantization mode: 'none' for bf16, 'nf4' for 4-bit NormalFloat quantization. Requires bitsandbytes."}),
                "cache_tag_embeds": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Pre-compute empty text embedding and unload text encoder to save VRAM"}),
                "group_offload": ("BOOLEAN", {"default": False,
                                               "tooltip": "Enable group offload to reduce peak VRAM at cost of slower speed"}),
                "auto_download": ("BOOLEAN", {"default": True,
                                               "tooltip": "If model is not found locally, download from HuggingFace. Disable to force local-only and error out instead of downloading."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_DEPTH_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "load_model"
    CATEGORY = "SeeThrough"

    def load_model(self, model, quant_mode="none", cache_tag_embeds=True, group_offload=False, auto_download=True):
        use_nf4 = quant_mode == "nf4"
        dtype = _pick_dtype(use_nf4)

        if use_nf4 and model in _NF4_REPO_MAP:
            model = _NF4_REPO_MAP[model]
            print(f"[SeeThrough] quant_mode=nf4: auto-switched to NF4 repo {model}", flush=True)

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)
        local_only = is_local or not auto_download

        print(f"[SeeThrough] Loading Marigold depth model from: {pretrained} "
              f"(quant_mode={quant_mode}, local={is_local}, local_files_only={local_only})", flush=True)
        unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder="unet", local_files_only=local_only)
        pipeline = MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet, local_files_only=local_only)

        if use_nf4:
            print(f"[SeeThrough] NF4 mode: casting non-quantized parameters to {dtype}", flush=True)
            pipeline.vae.to(dtype=dtype)
            _cast_non_quantized_params(pipeline.unet, dtype)
            _cast_non_quantized_params(pipeline.text_encoder, dtype)
        else:
            pipeline.to(dtype=dtype)

        pipeline._st_group_offload = False
        if group_offload and use_nf4:
            # Group offload moves module blocks on and off the GPU with hooks, but a
            # bitsandbytes Params4bit carries a quant_state holding device pointers that
            # the hook's .to() does not migrate. The next kernel then dereferences a stale
            # pointer and the process dies with "CUDA error: an illegal memory access was
            # encountered" — an abort, not a catchable OOM. NF4 already cuts VRAM enough
            # that offloading on top of it buys little, so quantization wins.
            print("[SeeThrough] WARNING: group_offload is incompatible with quant_mode=nf4 "
                  "(bitsandbytes quant_state is not migrated by the offload hooks, which "
                  "crashes with an illegal memory access). Disabling group_offload; "
                  "NF4 alone is the lower-VRAM option.", flush=True)
            group_offload = False

        if group_offload:
            if hasattr(pipeline, 'enable_group_offload'):
                print("[SeeThrough] Enabling group offload for Marigold pipeline", flush=True)
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
                pipeline._st_group_offload = True
            else:
                print("[SeeThrough] WARNING: group_offload requires diffusers >= 0.37.0, skipping. Please upgrade: pip install diffusers>=0.37.0", flush=True)

        if cache_tag_embeds:
            if not pipeline._st_group_offload:
                device = mm.get_torch_device()
                if not use_nf4:
                    pipeline.text_encoder.to(device)
            print("[SeeThrough] Caching empty text embedding and unloading text encoder...", flush=True)
            pipeline.cache_tag_embeds(unload_textencoders=True)
            _log_vram("After Marigold cache_tag_embeds (text encoder unloaded)")

        _log_vram("Depth model loaded (CPU)")
        print(f"[SeeThrough] Depth model loaded (quant_mode={quant_mode})", flush=True)
        return (pipeline,)

class SeeThrough_GenerateLayers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "resolution": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERS", "IMAGE")
    RETURN_NAMES = ("layers", "preview")
    FUNCTION = "generate"
    CATEGORY = "SeeThrough"

    def generate(self, image, layerdiff_model, seed=42, resolution=1280, num_inference_steps=30):
        pipeline = layerdiff_model
        device = mm.get_torch_device()
        offload = torch.device("cpu")
        seed_everything(seed)

        # Convert ComfyUI IMAGE to numpy RGBA
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 3:
            img_np = np.concatenate([img_np, np.full((*img_np.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        input_img = img_np.copy()

        fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
        scale = pad_size[0] / resolution

        tag_version = pipeline.unet.get_tag_version()
        layer_dict = {}

        print(f"[SeeThrough] GenerateLayers: tag_version={tag_version}, resolution={resolution}, steps={num_inference_steps}", flush=True)
        _log_vram("GenerateLayers start")

        is_group_offload = getattr(pipeline, '_st_group_offload', False)

        has_cached = len(pipeline._cached_prompt_embeds) > 0
        if has_cached:
            print("[SeeThrough] Using cached tag embeddings", flush=True)
            encode_fn = pipeline.encode_cropped_prompt_77tokens_cached
        else:
            if not is_group_offload:
                pipeline.text_encoder.to(device)
                pipeline.text_encoder_2.to(device)
                _log_vram("Text encoders loaded to GPU")
            encode_fn = pipeline.encode_cropped_prompt_77tokens

        if tag_version == "v2":
            prompt_embeds, pooled_prompt_embeds = encode_fn(VALID_BODY_PARTS_V2)
        elif tag_version == "v3":
            body_tags = ["front hair", "back hair", "head", "neck", "neckwear",
                         "topwear", "handwear", "bottomwear", "legwear", "footwear",
                         "tail", "wings", "objects"]
            head_tags = ["headwear", "face", "irides", "eyebrow", "eyewhite",
                         "eyelash", "eyewear", "ears", "earwear", "nose", "mouth"]
            body_embeds, body_pooled = encode_fn(body_tags)
            head_embeds, head_pooled = encode_fn(head_tags)
        else:
            raise ValueError(f"Unknown tag version: {tag_version}")

        if not has_cached and not is_group_offload:
            pipeline.text_encoder.to(offload)
            pipeline.text_encoder_2.to(offload)
            _log_vram("Text encoders offloaded to CPU")

        if not is_group_offload:
            pipeline.unet.to(device)
            pipeline.vae.to(device)
        pipeline.trans_vae.to(device)
        mm.soft_empty_cache()
        _log_vram("UNet+VAE on GPU, ready for diffusion")

        rng = torch.Generator(device=device).manual_seed(seed)

        print("[SeeThrough] Text encoded, text encoders offloaded to CPU", flush=True)

        if tag_version == "v2":
            out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                           generator=rng, guidance_scale=1.0,
                           prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                           fullpage=fullpage)
            _log_vram("v2 diffusion complete")
            for rst, tag in zip(out.images, VALID_BODY_PARTS_V2):
                layer_dict[tag] = rst

        elif tag_version == "v3":
            out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                           generator=rng, guidance_scale=1.0,
                           prompt_embeds=body_embeds, pooled_prompt_embeds=body_pooled,
                           fullpage=fullpage, group_index=0)
            _log_vram("v3 body diffusion complete")
            for rst, tag in zip(out.images, body_tags):
                layer_dict[tag] = rst

            head_img = out.images[2]
            nz = cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
            if nz is not None:
                hx0, hy0, hw, hh = cv2.boundingRect(nz)
                hx = int(hx0 * scale) - pad_pos[0]
                hy = int(hy0 * scale) - pad_pos[1]
                input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, int(hw * scale), int(hh * scale)])
                hx1 = int(hx1 / scale + pad_pos[0] / scale)
                hy1 = int(hy1 / scale + pad_pos[1] / scale)
                ih, iw = input_head.shape[:2]
                input_head, head_pad_size, head_pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

                out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                               generator=rng, guidance_scale=1.0,
                               prompt_embeds=head_embeds, pooled_prompt_embeds=head_pooled,
                               fullpage=input_head, group_index=1)
                _log_vram("v3 head diffusion complete")

                canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
                coords = np.array([head_pad_pos[1], head_pad_pos[1] + ih, head_pad_pos[0], head_pad_pos[0] + iw])
                py1, py2, px1, px2 = (coords / scale).astype(np.int64)
                scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))

                for rst, tag in zip(out.images, head_tags):
                    rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
                    full = canvas.copy()
                    full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
                    layer_dict[tag] = full

        if not is_group_offload:
            pipeline.unet.to(offload)
            pipeline.vae.to(offload)
        pipeline.trans_vae.to(offload)
        mm.soft_empty_cache()
        _log_vram("GenerateLayers offloaded to CPU")
        print(f"[SeeThrough] GenerateLayers complete: {len(layer_dict)} layers", flush=True)

        layers_data = SeeThrough_LayersData(layer_dict, fullpage, input_img, resolution, pad_size, pad_pos)

        preview_dict = {}
        for tag, img in layer_dict.items():
            mask = img[..., -1] > 10
            if np.any(mask):
                preview_dict[tag] = {"img": img, "xyxy": [0, 0, resolution, resolution]}
        preview = _make_preview(preview_dict, resolution)

        return (layers_data, preview)


class SeeThrough_GenerateDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers": ("SEETHROUGH_LAYERS",),
                "depth_model": ("SEETHROUGH_DEPTH_MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "resolution_depth": ("INT", {"default": -1, "min": -1, "max": 2048, "step": 64,
                                              "tooltip": "Resolution for depth inference. -1 uses the same resolution as layers. Lower values save VRAM and speed up inference."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERS_DEPTH", "IMAGE")
    RETURN_NAMES = ("layers_depth", "preview")
    FUNCTION = "generate"
    CATEGORY = "SeeThrough"

    def generate(self, layers, depth_model, seed=42, resolution_depth=-1):
        layer_dict = layers.layer_dict
        fullpage = layers.fullpage
        resolution = layers.resolution
        marigold = depth_model
        device = mm.get_torch_device()
        offload = torch.device("cpu")
        is_group_offload = getattr(marigold, '_st_group_offload', False)

        print("[SeeThrough] GenerateDepth: running Marigold...", flush=True)
        _log_vram("GenerateDepth start")

        empty_array = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        blended_alpha = np.zeros((resolution, resolution), dtype=np.float32)
        compose_list = {"eyes": ["eyewhite", "irides", "eyelash", "eyebrow"],
                        "hair": ["back hair", "front hair"]}

        img_list = []
        for tag in VALID_BODY_PARTS_V2:
            if tag in layer_dict:
                tag_arr = layer_dict[tag].copy()
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                img_list.append(tag_arr)
            else:
                img_list.append(empty_array.copy())

        compose_dict = {}
        for c, clist in compose_list.items():
            imlist, taglist = [], []
            for t in clist:
                if t in layer_dict:
                    tag_arr = layer_dict[t].copy()
                    tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                    imlist.append(tag_arr)
                    taglist.append(t)
            if imlist:
                composed = img_alpha_blending(imlist, premultiplied=False)
                img_list[VALID_BODY_PARTS_V2.index(c)] = composed
                compose_dict[c] = {"taglist": taglist, "imlist": imlist}

        for img in img_list:
            blended_alpha += img[..., -1].astype(np.float32) / 255
        blended_alpha = np.clip(blended_alpha, 0, 1) * 255
        blended_alpha = blended_alpha.astype(np.uint8)

        fullpage_for_depth = fullpage.copy()
        fullpage_for_depth[..., -1] = blended_alpha
        img_list.append(fullpage_for_depth)

        src_h, src_w = resolution, resolution
        if resolution_depth > 0 and resolution_depth != resolution:
            depth_res = resolution_depth
            depth_res = max(64, (depth_res // 8) * 8)
            img_list_input = [smart_resize(img, (depth_res, depth_res)) for img in img_list]
            print(f"[SeeThrough] Depth inference at resolution {depth_res} (layers at {resolution})", flush=True)
        else:
            img_list_input = img_list
            depth_res = resolution

        if not is_group_offload:
            marigold.unet.to(device)
            marigold.vae.to(device)
            mm.soft_empty_cache()
            _log_vram("Marigold on GPU")
            print("[SeeThrough] Marigold pipeline moved to GPU", flush=True)

        seed_everything(seed)
        pipe_out = marigold(color_map=None, show_progress_bar=False, img_list=img_list_input)
        _log_vram("Marigold inference complete")
        depth_pred = pipe_out.depth_tensor.to(device="cpu", dtype=torch.float32).numpy()

        if depth_res != resolution:
            depth_pred = np.stack([smart_resize(d, (src_h, src_w)) for d in depth_pred])

        if not is_group_offload:
            marigold.unet.to(offload)
            marigold.vae.to(offload)
            mm.soft_empty_cache()
            _log_vram("GenerateDepth offloaded to CPU")

        depth_dict = {}
        for ii, tag in enumerate(VALID_BODY_PARTS_V2):
            depth = depth_pred[ii]
            if tag in compose_dict:
                mask_accum = blended_alpha > 256  # all-False
                for t, im in zip(compose_dict[tag]["taglist"][::-1], compose_dict[tag]["imlist"][::-1]):
                    mask_local = im[..., -1] > 15
                    mask_invis = np.bitwise_and(mask_accum, mask_local)
                    depth_local = np.full((resolution, resolution), fill_value=1.0, dtype=np.float32)
                    depth_local[mask_local] = depth[mask_local]
                    if np.any(mask_invis):
                        vis = np.bitwise_and(mask_local, np.bitwise_not(mask_invis))
                        if np.any(vis):
                            depth_local[mask_invis] = np.median(depth[vis])
                    mask_accum = np.bitwise_or(mask_accum, mask_local)
                    depth_dict[t] = depth_local
            else:
                depth_dict[tag] = np.clip(depth, 0, 1).astype(np.float32)

        print(f"[SeeThrough] GenerateDepth complete: {len(depth_dict)} depth maps, Marigold offloaded to CPU", flush=True)

        result = SeeThrough_LayersDepthData(layer_dict, depth_dict, fullpage, resolution)

        # Preview: blend with depth info
        preview_dict = {}
        for tag in layer_dict:
            img = layer_dict[tag]
            if tag in depth_dict and np.any(img[..., -1] > 10):
                preview_dict[tag] = {"img": img, "depth": depth_dict[tag], "xyxy": [0, 0, resolution, resolution]}
        preview = _make_preview(preview_dict, resolution)

        return (result, preview)

class SeeThrough_PostProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers_depth": ("SEETHROUGH_LAYERS_DEPTH",),
                "tblr_split": ("BOOLEAN", {"default": True,
                                           "tooltip": "Split symmetric parts (eyes, ears, handwear) into left/right"}),
                "use_lama": ("BOOLEAN", {"default": True,
                                         "tooltip": "Use LaMa inpainting for hair splitting (better quality). Falls back to OpenCV if disabled."}),
            },
            "optional": {
                "split_hair": ("BOOLEAN", {"default": True,
                                           "tooltip": "v2 models only: split the single 'hair' layer into hairf/hairb by depth. "
                                                      "Disable to keep 'hair' whole and cut it with SeeThrough Split Layer instead."}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_PARTS", "IMAGE")
    RETURN_NAMES = ("parts", "preview")
    FUNCTION = "process"
    CATEGORY = "SeeThrough"

    def process(self, layers_depth, tblr_split=True, use_lama=True, split_hair=True):
        layer_dict = layers_depth.layer_dict
        depth_dict = layers_depth.depth_dict
        fullpage = layers_depth.fullpage
        resolution = layers_depth.resolution

        print("[SeeThrough] PostProcess: splitting & clustering...", flush=True)

        # Build tag2pinfo
        tag2pinfo = {}
        for tag in layer_dict:
            img = layer_dict[tag]
            if tag not in depth_dict:
                continue
            depth = depth_dict[tag]
            mask = img[..., -1] > 10
            if not np.any(mask):
                continue
            tag2pinfo[tag] = {"img": img, "depth": depth, "xyxy": [0, 0, resolution, resolution],
                              "mask": mask, "tag": tag}

        # Eye splitting (v2 composite 'eyes')
        if "eyes" in tag2pinfo:
            part_info = tag2pinfo.pop("eyes")
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                part_info["mask"].astype(np.uint8) * 255, connectivity=8)
            if len(stats) > 2:
                stats_arr = np.array(stats)
                if len(stats_arr[..., -1]) >= 5:
                    stats_order = np.argsort(stats_arr[..., -1])[::-1][1:]
                    eyel_mask, eyer_mask, statsl, statsr = _label_lr_split(labels, stats_arr, stats_order[0], stats_order[1])
                    img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl)
                    tag2pinfo["eyer"] = {"img": img, "xyxy": xyxy, "depth": depth}
                    img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr)
                    tag2pinfo["eyel"] = {"img": img, "xyxy": xyxy, "depth": depth}
                    if len(stats_order) >= 4:
                        browl_mask, browr_mask, statsl, statsr = _label_lr_split(labels, stats_arr, stats_order[2], stats_order[3])
                        img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsl)
                        tag2pinfo["browr"] = {"img": img, "xyxy": xyxy, "depth": depth}
                        img, depth, xyxy, _ = _process_cuts(part_info["img"], part_info["depth"], part_info["xyxy"], statsr)
                        tag2pinfo["browl"] = {"img": img, "xyxy": xyxy, "depth": depth}
                else:
                    tag2pinfo["eyes"] = part_info
            else:
                tag2pinfo["eyes"] = part_info

        # Left-right splitting
        if tblr_split:
            _tag_lr_split("handwear", tag2pinfo)
            for eye_tag in ["eyewhite", "irides", "eyelash", "eyebrow"]:
                _tag_lr_split(eye_tag, tag2pinfo)
            _tag_lr_split("ears", tag2pinfo)

            if split_hair and "hair" in tag2pinfo:
                part_info = tag2pinfo.pop("hair")
                try:
                    inpaint_mode = "lama" if use_lama else "cv2"
                    parts = cluster_inpaint_part(inpaint=inpaint_mode, **part_info)
                    parts.sort(key=lambda x: x["depth_median"])
                    tag2pinfo["hairf"] = parts[0]
                    tag2pinfo["hairb"] = parts[1]
                except Exception as e:
                    print(f"[SeeThrough] Hair clustering failed: {e}, keeping as-is", flush=True)
                    tag2pinfo["hair"] = part_info

        # Nose/mouth color restoration
        for restore_tag in ("nose", "mouth"):
            if restore_tag in tag2pinfo:
                pinfo = tag2pinfo[restore_tag]
                src_h, src_w = pinfo["img"].shape[:2]
                fp_h, fp_w = fullpage.shape[:2]
                if src_h == fp_h and src_w == fp_w:
                    pinfo["img"][..., :3] = fullpage[..., :3]
                else:
                    x1, y1 = pinfo["xyxy"][0], pinfo["xyxy"][1]
                    crop = fullpage[y1:min(y1 + src_h, fp_h), x1:min(x1 + src_w, fp_w), :3]
                    pinfo["img"][:crop.shape[0], :crop.shape[1], :3] = crop

        # Crop + depth_median
        for tag in list(tag2pinfo.keys()):
            pinfo = tag2pinfo[tag]
            if "img" in pinfo and "depth" in pinfo:
                _compute_depth_median(pinfo)
            pinfo["tag"] = tag

        # Depth ordering adjustments
        if "face" in tag2pinfo:
            face_dm = tag2pinfo["face"]["depth_median"]
            for t in ["nose", "mouth", "eyes", "eyel", "eyer"]:
                if t in tag2pinfo and tag2pinfo[t]["depth_median"] > face_dm:
                    tag2pinfo[t]["depth_median"] = face_dm - 0.001
            for t in ["earr", "earl", "ears"]:
                if t in tag2pinfo:
                    tag2pinfo[t]["depth_median"] = face_dm + 0.001

        frame_size = fullpage.shape[:2]
        parts_data = {"tag2pinfo": tag2pinfo, "frame_size": frame_size}

        print(f"[SeeThrough] PostProcess complete: {len(tag2pinfo)} layers", flush=True)
        for tag, pinfo in sorted(tag2pinfo.items(), key=lambda x: x[1].get("depth_median", 1)):
            dm = pinfo.get("depth_median", "?")
            print(f"  - {tag}: depth_median={dm:.4f}" if isinstance(dm, float) else f"  - {tag}", flush=True)

        preview = _make_preview(tag2pinfo, resolution)
        return (parts_data, preview)


class SeeThrough_SavePSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
                "filename_prefix": ("STRING", {"default": "seethrough"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_file",)
    FUNCTION = "save"
    CATEGORY = "SeeThrough"
    OUTPUT_NODE = True

    def save(self, parts, filename_prefix="seethrough"):
        from PIL import Image
        import json

        tag2pinfo = parts["tag2pinfo"]
        frame_size = parts["frame_size"]
        canvas_h, canvas_w = frame_size

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        sorted_tags = sorted(tag2pinfo.keys(), key=lambda t: tag2pinfo[t].get("depth_median", 1), reverse=True)

        layer_info_list = []
        for tag in sorted_tags:
            pinfo = tag2pinfo[tag]
            img = pinfo.get("img")
            depth = pinfo.get("depth")
            if img is None:
                continue

            xyxy = pinfo.get("xyxy", [0, 0, img.shape[1], img.shape[0]])
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            layer_filename = f"{filename_prefix}_{ts}_{uid}_{tag}.png"
            Image.fromarray(img).save(os.path.join(output_dir, layer_filename))

            entry = {"name": tag, "filename": layer_filename,
                     "left": x1, "top": y1, "right": x2, "bottom": y2,
                     "depth_median": float(pinfo.get("depth_median", 1))}

            if depth is not None:
                depth_filename = f"{filename_prefix}_{ts}_{uid}_{tag}_depth.png"
                if depth.ndim == 2:
                    Image.fromarray(depth, mode="L").save(os.path.join(output_dir, depth_filename))
                else:
                    Image.fromarray(depth).save(os.path.join(output_dir, depth_filename))
                entry["depth_filename"] = depth_filename

            layer_info_list.append(entry)

        info_filename = f"{filename_prefix}_{ts}_{uid}_layers.json"
        info_path = os.path.join(output_dir, info_filename)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"prefix": filename_prefix, "timestamp": f"{ts}_{uid}",
                       "layers": layer_info_list, "width": int(canvas_w), "height": int(canvas_h)}, f, indent=2)

        log_path = os.path.join(output_dir, "seethrough_psd_info.log")
        with open(log_path, "w") as f:
            f.write(info_filename)

        print(f"[SeeThrough] {len(layer_info_list)} layers saved. Use 'Download PSD' button to generate PSD.", flush=True)
        return (info_path,)


class SeeThrough_PartsToLayers:
    """Convert SEETHROUGH_PARTS into a core LAYERS document for the built-in
    compositor / layer editor (Create Layered Image node)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
            },
        }

    RETURN_TYPES = ("LAYERS",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "convert"
    CATEGORY = "SeeThrough"

    def convert(self, parts):
        tag2pinfo = parts["tag2pinfo"]
        canvas_h, canvas_w = parts["frame_size"]

        sorted_tags = sorted(tag2pinfo.keys(),
                             key=lambda t: tag2pinfo[t].get("depth_median", 1), reverse=True)

        items = []
        for z_index, tag in enumerate(sorted_tags):
            pinfo = tag2pinfo[tag]
            img = pinfo.get("img")
            if img is None:
                continue
            xyxy = pinfo.get("xyxy", [0, 0, img.shape[1], img.shape[0]])
            tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            items.append({
                "image": tensor,
                "type": "raster",
                "name": tag,
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "z_index": z_index,
            })

        document = {
            "version": 1,
            "layers": items,
            "canvas": (int(canvas_w), int(canvas_h)),
        }
        return (document,)


def _paste_canvas(canvas, img, xyxy):
    """Paste img (HxWxC) into canvas at xyxy, clipping to the canvas."""
    x1, y1 = int(xyxy[0]), int(xyxy[1])
    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]
    x2, y2 = min(x1 + w, cw), min(y1 + h, ch)
    if x2 <= x1 or y2 <= y1:
        return canvas
    canvas[y1:y2, x1:x2] = img[:y2 - y1, :x2 - x1]
    return canvas


def _merge_parts(pinfos):
    """Composite several cropped parts (front = smallest depth_median) into one RGBA + depth."""
    x1 = min(int(p["xyxy"][0]) for p in pinfos)
    y1 = min(int(p["xyxy"][1]) for p in pinfos)
    x2 = max(int(p["xyxy"][2]) for p in pinfos)
    y2 = max(int(p["xyxy"][3]) for p in pinfos)
    h, w = y2 - y1, x2 - x1
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    depth = np.ones((h, w), dtype=np.float32)
    # back to front, straight-alpha "over"
    for p in sorted(pinfos, key=lambda q: q.get("depth_median", 1.0), reverse=True):
        img = p["img"]
        ph, pw = img.shape[:2]
        ox, oy = int(p["xyxy"][0]) - x1, int(p["xyxy"][1]) - y1
        a = img[..., 3].astype(np.float32) / 255.0
        c = img[..., :3].astype(np.float32)
        dst_rgb = rgb[oy:oy + ph, ox:ox + pw]
        dst_a = alpha[oy:oy + ph, ox:ox + pw]
        out_a = a + dst_a * (1 - a)
        num = c * a[..., None] + dst_rgb * (dst_a * (1 - a))[..., None]
        dst_rgb[...] = np.where(out_a[..., None] > 1e-6, num / np.maximum(out_a, 1e-6)[..., None], dst_rgb)
        dst_a[...] = out_a
        pd = p["depth"]
        pd = pd.astype(np.float32) / 255.0 if pd.dtype == np.uint8 else pd.astype(np.float32)
        vis = img[..., 3] > 15
        depth[oy:oy + ph, ox:ox + pw][vis] = pd[vis]
    img = np.concatenate([np.round(rgb).astype(np.uint8), np.round(alpha * 255).astype(np.uint8)[..., None]], axis=-1)
    return img, depth, [x1, y1, x2, y2]


class SeeThrough_SplitLayer:
    """Cut one (or several merged) layers into K pieces, front to back, inpainting what
    each piece hides so the pieces behind it stay complete. Default use: hair pieces."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
                "tags": ("STRING", {"default": "front hair, back hair",
                                    "tooltip": "Layer name(s) to split, comma separated. Several names are merged first "
                                               "(v3 models: 'front hair, back hair'; v2 models: 'hair' with split_hair off)."}),
                "mode": (SPLIT_MODES, {"default": "lineart_watershed",
                                       "tooltip": "depth_position: KMeans on x,y,depth. lineart_watershed: same seeds, boundaries snapped to line art. "
                                                  "depth_kmeans: depth only (original front/back logic). masks: use the 'masks' input."}),
                "num_pieces": ("INT", {"default": 4, "min": 1, "max": 20,
                                       "tooltip": "Target number of pieces for the automatic modes (final count may differ after component splitting)."}),
                "output_prefix": ("STRING", {"default": "hair",
                                             "tooltip": "Pieces are named <prefix>-0, <prefix>-1, ... front to back."}),
                "inpaint": (["lama", "cv2"], {"default": "lama",
                                              "tooltip": "How to fill the area hidden behind each front piece."}),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "mode=masks: one mask per piece, canvas coordinates (use SeeThrough Layer To Image as the base)."}),
                "depth_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                           "tooltip": "How much depth counts vs. position in depth_position / lineart_watershed."}),
                "min_area_ratio": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005,
                                             "tooltip": "Pieces smaller than this fraction of the layer are merged into neighbours."}),
                "split_components": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Give disconnected regions of one cluster their own piece (e.g. left/right side locks)."}),
                "max_pieces": ("INT", {"default": 8, "min": 0, "max": 40,
                                       "tooltip": "Hard cap on the number of output pieces (smallest are merged into neighbours). 0 = no cap."}),
                "mask_order": (["depth", "input"], {"default": "depth",
                                                   "tooltip": "mode=masks: order pieces by depth, or keep the input mask order (first = front)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_PARTS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("parts", "preview", "pieces_preview")
    FUNCTION = "split"
    CATEGORY = "SeeThrough"

    def split(self, parts, tags, mode, num_pieces, output_prefix, inpaint, masks=None,
              depth_weight=1.0, min_area_ratio=0.02, split_components=True, max_pieces=8, mask_order="depth", seed=0):
        tag2pinfo = dict(parts["tag2pinfo"])
        frame_size = parts["frame_size"]
        canvas_h, canvas_w = int(frame_size[0]), int(frame_size[1])

        wanted = [t.strip() for t in tags.split(",") if t.strip()]
        found = [t for t in wanted if t in tag2pinfo and tag2pinfo[t].get("img") is not None]
        if not found:
            print(f"[SeeThrough] SplitLayer: none of {wanted} in parts {sorted(tag2pinfo.keys())}, passing through", flush=True)
            return (parts, _make_preview(tag2pinfo, canvas_h), torch.zeros((1, canvas_h, canvas_w, 3)))

        if len(found) == 1:
            src = tag2pinfo[found[0]]
            img = src["img"].copy()
            depth = src["depth"]
            depth = depth.astype(np.float32) / 255.0 if depth.dtype == np.uint8 else depth.astype(np.float32).copy()
            xyxy = [int(v) for v in src["xyxy"]]
        else:
            img, depth, xyxy = _merge_parts([tag2pinfo[t] for t in found])
        x1, y1, x2, y2 = xyxy
        mask = img[..., 3] > 10
        if not np.any(mask):
            return (parts, _make_preview(tag2pinfo, canvas_h), torch.zeros((1, canvas_h, canvas_w, 3)))

        mask_list = None
        if mode == "masks":
            if masks is None:
                raise ValueError("SeeThrough Split Layer: mode 'masks' needs the masks input")
            m_np = masks.detach().cpu().numpy()
            if m_np.ndim == 2:
                m_np = m_np[None]
            mask_list = []
            for m in m_np:
                if m.shape != (canvas_h, canvas_w):
                    m = cv2.resize(m.astype(np.float32), (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
                mask_list.append(m[y1:y2, x1:x2] > 0.5)

        labels, meds = compute_labels(img, depth, mask, mode=mode, k=num_pieces, depth_weight=depth_weight,
                                      min_area_ratio=min_area_ratio, split_components=split_components,
                                      masks=mask_list, order=mask_order, seed=seed,
                                      max_pieces=max_pieces if mode != "masks" else 0)
        n_labels = len(meds)
        print(f"[SeeThrough] SplitLayer: {found} -> {n_labels} pieces (mode={mode}, inpaint={inpaint})", flush=True)

        if n_labels <= 1:
            pieces = [{"img": img, "depth": depth, "depth_median": float(np.median(depth[mask]))}]
        else:
            try:
                pieces = split_part_by_labels(img, depth, labels, inpaint=inpaint)
            except Exception as e:
                if inpaint == "lama":
                    print(f"[SeeThrough] SplitLayer: LaMa failed ({e}), falling back to cv2 inpaint", flush=True)
                    pieces = split_part_by_labels(img, depth, labels, inpaint="cv2")
                else:
                    raise
            finally:
                if inpaint == "lama":
                    unload_lama()

        for t in found:
            tag2pinfo.pop(t, None)
        prefix = output_prefix.strip() or found[0]
        for i, piece in enumerate(pieces):
            name = f"{prefix}-{i}"
            piece["xyxy"] = list(xyxy)
            piece["tag"] = name
            _compute_depth_median(piece)
            tag2pinfo[name] = piece
            print(f"  - {name}: depth_median={piece['depth_median']:.4f}", flush=True)

        # label visualisation on a mid-grey canvas
        base = np.full((y2 - y1, x2 - x1, 3), 128, dtype=np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        base = np.round(img[..., :3].astype(np.float32) * a + base * (1 - a)).astype(np.uint8)
        overlay = label_overlay(base, labels)
        canvas = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
        canvas = _paste_canvas(canvas, overlay, xyxy)
        pieces_preview = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)

        new_parts = {"tag2pinfo": tag2pinfo, "frame_size": frame_size}
        return (new_parts, _make_preview(tag2pinfo, canvas_h), pieces_preview)


class SeeThrough_LayerToImage:
    """Render one layer at canvas size as IMAGE + MASK, e.g. to paint or SAM masks for Split Layer."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
                "tags": ("STRING", {"default": "front hair, back hair",
                                    "tooltip": "Layer name(s), comma separated; several are composited by depth."}),
                "background": (["gray", "white", "black"], {"default": "gray"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "layer_names")
    FUNCTION = "render"
    CATEGORY = "SeeThrough"

    def render(self, parts, tags, background="gray"):
        tag2pinfo = parts["tag2pinfo"]
        canvas_h, canvas_w = [int(v) for v in parts["frame_size"]]
        names = ", ".join(sorted(tag2pinfo.keys(), key=lambda t: tag2pinfo[t].get("depth_median", 1)))
        wanted = [t.strip() for t in tags.split(",") if t.strip()]
        found = [t for t in wanted if t in tag2pinfo and tag2pinfo[t].get("img") is not None]
        bg = {"gray": 128, "white": 255, "black": 0}[background]
        rgb = np.full((canvas_h, canvas_w, 3), bg, dtype=np.float32)
        alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        if found:
            img, _, xyxy = _merge_parts([tag2pinfo[t] for t in found])
            x1, y1 = xyxy[0], xyxy[1]
            h, w = img.shape[:2]
            a = img[..., 3].astype(np.float32) / 255.0
            rgb[y1:y1 + h, x1:x1 + w] = img[..., :3].astype(np.float32) * a[..., None] + rgb[y1:y1 + h, x1:x1 + w] * (1 - a)[..., None]
            alpha[y1:y1 + h, x1:x1 + w] = a
        else:
            print(f"[SeeThrough] LayerToImage: none of {wanted} found in [{names}]", flush=True)
        image = torch.from_numpy(rgb / 255.0).unsqueeze(0)
        mask = torch.from_numpy(alpha).unsqueeze(0)
        return (image, mask, names)


NODE_CLASS_MAPPINGS = {
    "SeeThrough_LoadLayerDiffModel": SeeThrough_LoadLayerDiffModel,
    "SeeThrough_LoadDepthModel": SeeThrough_LoadDepthModel,
    "SeeThrough_GenerateLayers": SeeThrough_GenerateLayers,
    "SeeThrough_GenerateDepth": SeeThrough_GenerateDepth,
    "SeeThrough_PostProcess": SeeThrough_PostProcess,
    "SeeThrough_SavePSD": SeeThrough_SavePSD,
    "SeeThrough_PartsToLayers": SeeThrough_PartsToLayers,
    "SeeThrough_SplitLayer": SeeThrough_SplitLayer,
    "SeeThrough_LayerToImage": SeeThrough_LayerToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeeThrough_LoadLayerDiffModel": "SeeThrough Load LayerDiff Model",
    "SeeThrough_LoadDepthModel": "SeeThrough Load Depth Model",
    "SeeThrough_GenerateLayers": "SeeThrough Generate Layers",
    "SeeThrough_GenerateDepth": "SeeThrough Generate Depth",
    "SeeThrough_PostProcess": "SeeThrough Post Process",
    "SeeThrough_SavePSD": "SeeThrough Save PSD",
    "SeeThrough_PartsToLayers": "SeeThrough Parts To Layers",
    "SeeThrough_SplitLayer": "SeeThrough Split Layer",
    "SeeThrough_LayerToImage": "SeeThrough Layer To Image",
}
