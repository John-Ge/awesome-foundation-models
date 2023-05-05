# Awesome-Foundation-Model-Papers

![](https://img.shields.io/badge/Papercount-50-green)


A library of foundation models in computer vision, natural language processing and multi-modal learning. This repo mainly include pretraining methods, foundation models, fine-tuning methods and some projects *etc.*

Contributions are welcome!

本项目是一个视觉，语言和多模态基础模型的仓库。主要包括预训练方法，基础模型，微调方法和成熟的项目等。

欢迎大家为项目贡献！

- [Awesome-Foundation-Model-Papers](#awesome-foundation-model-papers)
- [Computer Vision](#computer-vision)
  - [Pretraining](#pretraining)
  - [Generation](#generation)
  - [Unified Architecture for Vision](#unified-architecture-for-vision)
- [NLP Foundation Models](#nlp-foundation-models)
  - [Pretraining](#pretraining-1)
  - [Instruction Tuning](#instruction-tuning)
  - [Human Chat Foundation Models](#human-chat-foundation-models)
    - [Chinese Support](#chinese-support)
- [Multi-Modal Learning](#multi-modal-learning)
  - [Pretraining](#pretraining-2)
  - [Visual Chat Models](#visual-chat-models)
  - [Datasets](#datasets)
- [Contributions](#contributions)
- [Citation](#citation)


# Computer Vision

## Pretraining

1. EVA: Visual Representation Fantasies from BAAI. [[01-paper]](https://arxiv.org/abs/2211.07636) [[02-paper]](https://arxiv.org/abs/2303.11331) [[code]](https://github.com/baaivision/EVA)
2. Scaling Vision Transformers. [[paper]](https://arxiv.org/abs/2302.05442) [[code]](https://github.com/google-research/big_vision)
3. Scaling Vision Transformers to 22 Billion Parameters. [[paper]](https://arxiv.org/abs/2302.05442)
4. Segment Anything. [[paper]](https://ai.facebook.com/research/publications/segment-anything/) [[code]](https://github.com/facebookresearch/segment-anything) [[project]](https://segment-anything.com/)


## Generation
1. Deep Floyd -IF [[project]](https://deepfloyd.ai/deepfloyd-if)
2. Consistency Models. [[paper]](https://arxiv.org/abs/2303.01469) [[code]](https://github.com/openai/consistency_models)
3. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise. [[paper]](https://arxiv.org/abs/2208.09392) [[code]](https://github.com/arpitbansal297/Cold-Diffusion-Models)
4. Edit Anything. [[code]](https://github.com/sail-sg/EditAnything)
5. Scaling up GANs for Text-to-Image Synthesis. [[paper]](https://arxiv.org/abs/2303.05511) 


## Unified Architecture for Vision

1. Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks
2. Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks
3. SegGPT: Segmenting Everything In Context. [[paper]](https://arxiv.org/abs/2304.03284) [[code]](https://github.com/baaivision/painter)
4. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. [[paper]](https://arxiv.org/abs/2303.05499) [[code]](https://github.com/idea-research/groundingdino)
5. Segment Everything Everywhere All at Once. [[paper]](https://arxiv.org/abs/2304.06718) [[paper]](https://github.com/ux-decoder/segment-everything-everywhere-all-at-once)
6. X-Decoder: Generalized Decoding for Pixel, Image, and Language. [[paper]](https://arxiv.org/pdf/2212.11270.pdf) [[code]](https://github.com/microsoft/X-Decoder)
7. Unicorn 🦄 : Towards Grand Unification of Object Tracking. [[paper]](https://arxiv.org/abs/2207.07078) [[code]](https://github.com/MasterBin-IIAU/Unicorn)
8. Universal Instance Perception as Object Discovery and Retrieval. [[paper]](https://arxiv.org/abs/2303.06674) [[code]](https://github.com/MasterBin-IIAU/UNINEXT)
9. OneFormer: One Transformer to Rule Universal Image Segmentation. [[paper]](https://arxiv.org/abs/2211.06220) [[code]](https://github.com/SHI-Labs/OneFormer)
10. A Simple Framework for Open-Vocabulary Segmentation and Detection. [[paper]](https://arxiv.org/pdf/2303.08131.pdf) [[code]](https://github.com/IDEA-Research/OpenSeeD)
11. FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation. [[paper]](https://arxiv.org/pdf/2303.17225.pdf) [[code]](https://arxiv.org/pdf/2303.17225.pdf)


# NLP Foundation Models

## Pretraining

1. GPT: Improving language understanding by generative pre-training.
2. GPT-2: Language Models are Unsupervised Multitask Learners. [[paper]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
3. GPT-3: Language Models are Few-Shot Learners [[paper]](https://arxiv.org/pdf/2005.14165.pdf)
4. GPT-4. [[paper]](https://arxiv.org/abs/2303.08774)
5. LLaMA: Open and Efficient Foundation Language Models. [[paper]](https://arxiv.org/abs/2302.13971v1) [[code]](https://github.com/facebookresearch/llama)
6. Pythia: Interpreting Autoregressive Transformers Across Time and Scale. [[paper]](https://arxiv.org/pdf/2304.01373.pdf) [[code]](https://github.com/EleutherAI/pythia)

## Instruction Tuning
1. InstructGPT: Training language models to follow instructions with human feedback. [[paper]](https://arxiv.org/pdf/2203.02155.pdf)
2. RRHF: Rank Responses to Align Language Models with Human Feedback without tears. [[paper]]() [[code]](https://link.zhihu.com/?target=https%3A//github.com/GanjinZero/RRHF) [[blog]](https://zhuanlan.zhihu.com/p/623382893)
3. LLaVA: Large Language and Vision Assistant. [[paper]](https://arxiv.org/abs/2304.08485) [[project]](https://llava-vl.github.io/) [[blog]](https://zhuanlan.zhihu.com/p/622907299)

## Human Chat Foundation Models

1. Stanford Alpaca: An Instruction-following LLaMA Model. [[code]](https://github.com/tatsu-lab/stanford_alpaca)
2. Alpaca LoRA. [[code]](https://github.com/tloen/alpaca-lora)
3. Vicuna. [[code]](https://github.com/lm-sys/FastChat)
4. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. [[code]](https://github.com/ZrrSkywalker/LLaMA-Adapter) [[paper]](https://arxiv.org/pdf/2303.16199.pdf) [[v2-paper]](https://arxiv.org/pdf/2304.15010.pdf)
5. Stable Vicuna [[project]](https://stability.ai/blog/stablevicuna-open-source-rlhf-chatbot)
6. Koala: A Dialogue Model for Academic Research. [[paper]](https://bair.berkeley.edu/blog/2023/04/03/koala/) [[code]](https://github.com/young-geng/EasyLM)
7. Open-Assistant. [[project]](https://github.com/LAION-AI/Open-Assistant)


### Chinese Support
1. MOSS [[code]](https://github.com/OpenLMLab/MOSS)
2. Luotuo [[code]](https://github.com/LC1332/Luotuo-Chinese-LLM)
3. Linly [[code]](https://github.com/CVI-SZU/Linly) [[blog]](https://zhuanlan.zhihu.com/p/625786369)
4. FastChat-T5. [[code]](https://github.com/lm-sys/FastChat)
5. ChatGLM-6B. [[code]](https://github.com/THUDM/ChatGLM-6B)
6. Chat-RWKV. [[code]](https://github.com/BlinkDL/RWKV-LM)
7. baize. [[paper]](https://arxiv.org/abs/2304.01196) [[code]](https://github.com/BlinkDL/RWKV-LM)

# Multi-Modal Learning

## Pretraining
1. Learning Transferable Visual Models From Natural Language Supervision. [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP)
2. Align before Fuse: Vision and Language Representation Learning with Momentum Distillation. [[paper]](https://arxiv.org/abs/2107.07651) [[code]](https://github.com/salesforce/ALBEF)
3. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. [[paper]](https://arxiv.org/abs/2201.12086) [[code]](https://github.com/salesforce/BLIP)
4. mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality [[paper]](https://arxiv.org/abs/2304.14178) [[code]](https://arxiv.org/abs/2304.14178) [[dome]](https://arxiv.org/abs/2304.14178) [[blog]](https://zhuanlan.zhihu.com/p/625631667)
5. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models [[code]](https://link.zhihu.com/?target=https%3A//github.com/salesforce/LAVIS/tree/main/projects/blip2)
6. Language Is Not All You Need: Aligning Perception with Language Models [[code]](https://link.zhihu.com/?target=https%3A//github.com/microsoft/unilm)
7. Versatile Diffusion: Text, Images and Variations All in One Diffusion Model [[code]](https://link.zhihu.com/?target=https%3A//github.com/SHI-Labs/Versatile-Diffusion)
8. LLaVA: Large Language and Vision Assistant. [[paper]](https://arxiv.org/abs/2304.08485) [[project]](https://llava-vl.github.io/) [[blog]](https://zhuanlan.zhihu.com/p/622907299)
9. PaLM-E: An Embodied Multimodal Language Model. [[paper]](https://arxiv.org/abs/2303.03378) [[code]](https://palm-e.github.io/)


## Visual Chat Models
1. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. [[paper]](http://arxiv.org/abs/2304.10592) [[code]](https://minigpt-4.github.io/)


## Datasets

1. DataComp: In search of the next generation of multimodal datasets. [[paper]](https://arxiv.org/pdf/2304.14108.pdf) [[project]](https://www.datacomp.ai.)


# Contributions

Contributions are welcome! Anyone interested in this program could send pull requests. I may list you as a contributor in this repo.

欢迎大家提交 pull request 来更新这个项目~我会将你列为项目的贡献者。

![](assets/foundation.png)


# Citation

Please cite the repo if you find it useful.

```bibtex
@misc{chunjiang2023tobeawesome,
  author={Chunjiang Ge},
  title = {Awesome-Foundation-Model-Papers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/John-Ge/awesome-foundation-models}},
}
```