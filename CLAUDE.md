# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3 (Segment Anything Model 3) is Meta's unified foundation model for promptable segmentation in images and videos. It detects, segments, and tracks objects using text or visual prompts (points, boxes, masks). Key capabilities include open-vocabulary concept segmentation with 270K+ unique concepts.

## Development Commands

```bash
# Installation (via uv from root workspace)
uv sync --group sam3

# Installation for running notebooks
uv sync --group sam3-notebooks

# Code formatting (SAM3 uses ufmt, not ruff)
uv run ufmt format .

# Training
uv run python sam3/train/train.py -c configs/<config>.yaml [--use-cluster 0|1]
```

## Architecture

### Core Components

**Model Package** (`sam3/model/`): Neural network architecture
- `sam3_image.py`, `sam3_image_processor.py` - Image segmentation with `Sam3Processor`
- `sam3_video_predictor.py`, `sam3_video_base.py` - Video segmentation with session-based API
- `encoder.py`, `decoder.py` - Transformer encoder-decoder
- `vitdet.py` - Vision Transformer backbone
- `maskformer_segmentation.py` - MaskFormer segmentation head
- `memory.py` - Temporal memory blocks for video
- `geometry_encoders.py` - Point/box/mask prompt encoding
- `text_encoder_ve.py` - Text encoder for vocabulary expansion

**SAM Core** (`sam3/sam/`): Foundational components from SAM2
- `transformer.py` - RoPE (Rotary Position Embedding) attention
- `mask_decoder.py`, `prompt_encoder.py` - Mask decoding and prompt encoding

**Training** (`sam3/train/`): Hydra-based training pipeline
- `train.py` - Entry point with Hydra config management
- `trainer.py` - Distributed training loop
- `data/` - Dataset implementations (`sam3_image_dataset.py`, `sam3_video_dataset.py`)
- `loss/loss_fns.py` - Loss functions
- `matcher.py` - Hungarian matcher for assignment

**Agent** (`sam3/agent/`): LLM-based reasoning for complex segmentation
- `agent_core.py` - Core agent with LLM integration
- `client_sam3.py`, `client_llm.py` - Model and LLM clients

**Evaluation** (`sam3/eval/`): Benchmark evaluation
- `cgf1_eval.py` - Concept-grounded F1 evaluation
- `coco_eval.py`, `ytvis_eval.py` - COCO and video instance segmentation
- `saco_veval_eval.py` - SA-Co video evaluation

**Performance** (`sam3/perflib/`): Optimized utilities
- `connected_components.py`, `nms.py`, `masks_ops.py` - GPU-optimized operations
- `triton/` - Triton kernel implementations

### Key Patterns

**Model Building**: Use `model_builder.py` factory functions:
```python
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
```

**Image Inference Flow**:
```
Image -> Sam3Processor.set_image() -> set_text_prompt()/set_box_prompt() -> masks, boxes, scores
```

**Video Inference Flow**: Session-based API
```python
video_predictor.handle_request({"type": "start_session", "resource_path": path})
video_predictor.handle_request({"type": "add_prompt", "session_id": id, "frame_index": 0, "text": prompt})
```

**Configuration**: Hydra YAML configs in `sam3/train/configs/`

## Prerequisites

- Python 3.12+
- PyTorch 2.7+ with CUDA 12.6+
- HuggingFace authentication for checkpoint access (`hf auth login`)
