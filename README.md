# Awesome-Foundation-Model-Papers

![](https://img.shields.io/badge/Papercount-100+-green)

A library of foundation models in computer vision, natural language processing and multi-modal learning. This repo mainly include pretraining methods, foundation models, fine-tuning methods and some projects *etc.*

Contributions are welcome!

本项目是一个视觉，语言和多模态基础模型的仓库。主要包括预训练方法，基础模型，微调方法和成熟的项目等。未来计划整理一些可以使用的开源模型和数据资源。

欢迎大家为项目贡献！

- [Awesome-Foundation-Model-Papers](#awesome-foundation-model-papers)
- [Computer Vision](#computer-vision)
  - [Pretraining](#pretraining)
  - [Generation](#generation)
  - [Unified Architecture for Vision](#unified-architecture-for-vision)
- [NLP Foundation Models](#nlp-foundation-models)
  - [Pretraining](#pretraining-1)
  - [Instruction Tuning](#instruction-tuning)
  - [RLHF](#rlhf)
  - [Chat Models](#chat-models)
    - [Chinese Support](#chinese-support)
- [Multi-Modal Learning](#multi-modal-learning)
  - [Pretraining](#pretraining-2)
  - [Visual Chat Models](#visual-chat-models)
  - [Datasets](#datasets)
  - [Evaluation](#evaluation)
  - [](#)
- [Contributions](#contributions)
- [Citation](#citation)

# Computer Vision
## Pretraining

1. MAE: Masked Autoencoders Are Scalable Vision Learners. [[paper]](https://arxiv.org/abs/2111.06377) [code](Masked Autoencoders Are Scalable Vision Learners)
2. EVA: Visual Representation Fantasies from BAAI. [[01-paper]](https://arxiv.org/abs/2211.07636) [[02-paper]](https://arxiv.org/abs/2303.11331) [[code]](https://github.com/baaivision/EVA)
3. Scaling Vision Transformers. [[paper]](https://arxiv.org/abs/2302.05442) [[code]](https://github.com/google-research/big_vision)
4.  Scaling Vision Transformers to 22 Billion Parameters. [[paper]](https://arxiv.org/abs/2302.05442)
5.  Segment Anything. [[paper]](https://ai.facebook.com/research/publications/segment-anything/) [[code]](https://github.com/facebookresearch/segment-anything) [[project]](https://segment-anything.com/)
6.  UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer. [[paper]](https://arxiv.org/abs/2211.09552) [[code]](https://github.com/OpenGVLab/UniFormerV2)

## Generation

1. Deep Floyd -IF [[project]](https://deepfloyd.ai/deepfloyd-if)
2. Consistency Models. [[paper]](https://arxiv.org/abs/2303.01469) [[code]](https://github.com/openai/consistency_models)
3. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise. [[paper]](https://arxiv.org/abs/2208.09392) [[code]](https://github.com/arpitbansal297/Cold-Diffusion-Models)
4. Edit Anything. [[code]](https://github.com/sail-sg/EditAnything)
5. GigaGAN: Scaling up GANs for Text-to-Image Synthesis. [[paper]](https://arxiv.org/abs/2303.05511) 
6. Parti: Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. [[paper]](https://arxiv.org/abs/2206.10789) [[project]](https://arxiv.org/abs/2206.10789)

## Unified Architecture for Vision

1. Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks
2. Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks
3. SegGPT: Segmenting Everything In Context. [[paper]](https://arxiv.org/abs/2304.03284) [[code]](https://github.com/baaivision/painter)
4. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. [[paper]](https://arxiv.org/abs/2303.05499) [[code]](https://github.com/idea-research/groundingdino)
5. SAM: Segment Everything Everywhere All at Once. [[paper]](https://arxiv.org/abs/2304.06718) [[paper]](https://github.com/ux-decoder/segment-everything-everywhere-all-at-once)
6. X-Decoder: Generalized Decoding for Pixel, Image, and Language. [[paper]](https://arxiv.org/pdf/2212.11270.pdf) [[code]](https://github.com/microsoft/X-Decoder)
7. Unicorn 🦄 : Towards Grand Unification of Object Tracking. [[paper]](https://arxiv.org/abs/2207.07078) [[code]](https://github.com/MasterBin-IIAU/Unicorn)
8. UniNeXt: Universal Instance Perception as Object Discovery and Retrieval. [[paper]](https://arxiv.org/abs/2303.06674) [[code]](https://github.com/MasterBin-IIAU/UNINEXT)
9. OneFormer: One Transformer to Rule Universal Image Segmentation. [[paper]](https://arxiv.org/abs/2211.06220) [[code]](https://github.com/SHI-Labs/OneFormer)
10. OpenSeeD: A Simple Framework for Open-Vocabulary Segmentation and Detection. [[paper]](https://arxiv.org/pdf/2303.08131.pdf) [[code]](https://github.com/IDEA-Research/OpenSeeD)
11. FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation. [[paper]](https://arxiv.org/pdf/2303.17225.pdf) [[code]](https://arxiv.org/pdf/2303.17225.pdf)
12. Pix2seq: A language modeling framework for object detection. [[v1-paper]](https://arxiv.org/abs/2109.10852) [[v2-paper]](https://openreview.net/pdf?id=tjFaqsSK2I3) [[code]](https://github.com/google-research/pix2seq)
13. TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding. [[paper]](https://openreview.net/forum?id=-CwPopPJda) [[supplementary]](https://arxiv.org/pdf/2304.00971.pdf) [[code]](https://github.com/prismformore/Multi-Task-Transformer)
14. Musketeer (All for One, and One for All): A Generalist Vision-Language Model with Task Explanation Prompts. [[paper]](https://arxiv.org/pdf/2305.07019.pdf)
15. Fast Segment Anything. [[paper]](http://arxiv.org/abs/2306.12156) [[code]](http://arxiv.org/abs/2306.12156)

# NLP Foundation Models

## Pretraining

1. GPT: Improving language understanding by generative pre-training.
2. GPT-2: Language Models are Unsupervised Multitask Learners. [[paper]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
3. GPT-3: Language Models are Few-Shot Learners [[paper]](https://arxiv.org/pdf/2005.14165.pdf)
4. GPT-4. [[paper]](https://arxiv.org/abs/2303.08774)
5. LLaMA: Open and Efficient Foundation Language Models. [[paper]](https://arxiv.org/abs/2302.13971v1) [[code]](https://github.com/facebookresearch/llama)
6. Pythia: Interpreting Autoregressive Transformers Across Time and Scale. [[paper]](https://arxiv.org/pdf/2304.01373.pdf) [[code]](https://github.com/EleutherAI/pythia)
7. PaLM： Scaling Language Modeling with Pathways. [[paper]](https://arxiv.org/abs/2204.02311)
8. RedPajama. [[blog]](https://link.zhihu.com/?target=https%3A//www.together.xyz/blog/redpajama-models-v1)
9. LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instruction [[paper]](https://arxiv.org/pdf/2304.14402v1.pdf) [[code]](https://github.com/mbzuai-nlp/LaMini-LM)
10. MPT. [[blog]](https://www.mosaicml.com/blog/mpt-7b) [[code]](https://github.com/mosaicml/llm-foundry/)
11. BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability. [[paper]](https://github.com/Neutralzz/BiLLa)
12. OpenLLaMA: An Open Reproduction of LLaMA. [[code]](https://github.com/openlm-research/open_llama)
13. InternLM. [[code]](https://github.com/InternLM/InternLM)

## Instruction Tuning

1. InstructGPT: Training language models to follow instructions with human feedback. [[paper]](https://arxiv.org/pdf/2203.02155.pdf) [[blog]](https://openai.com/research/instruction-following)
2. Principle-Driven Self-Alignment of Language Modelsfrom Scratch with Minimal Human Supervision. [[paper]](https://arxiv.org/abs/2305.03047) [[code]](https://mitibmdemos.draco.res.ibm.com/dromedary)
3. Scaling instruction-finetuned language models. [[paper]](https://arxiv.org/pdf/2210.11416.pdf) 
4. Self-Instruct: Aligning Language Model with Self Generated Instructions. [[paper]](https://arxiv.org/abs/2212.10560) [[code]](https://github.com/yizhongw/self-instruct)
5. LIMA: Less Is More for Alignment. [[paper]](https://arxiv.org/pdf/2305.11206.pdf)
6.  Orca: Progressive Learning from Complex Explanation Traces of GPT-4. [[paper]](https://arxiv.org/abs/2306.02707)
7.  WizardLM: An Instruction-following LLM Using Evol-Instruct. [[paper]](https://arxiv.org/abs/2304.12244) [[code]](https://github.com/nlpxucan/WizardLM)
8.  QLoRA: Efficient Finetuning of Quantized LLMs. [[paper]](https://arxiv.org/abs/2305.14314) [[code]](https://github.com/artidoro/qlora)
9.  Instruction Tuning with GPT-4. [[paper]](https://arxiv.org/abs/2304.03277) [[code]](https://arxiv.org/abs/2304.03277)

## RLHF

1. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[paper]](https://arxiv.org/abs/2204.05862) [[code]](https://github.com/anthropics/hh-rlhf)
2. RRHF: Rank Responses to Align Language Models with Human Feedback without tears. [[paper]](https://arxiv.org/pdf/2304.05302v1.pdf) [[code]](https://github.com/GanjinZero/RRHF) [[blog]](https://zhuanlan.zhihu.com/p/623382893)
3. Beaver. [[code]](https://panda-gpt.github.io/)
4. MOSS-RLHF. [[code]](https://openlmlab.github.io/MOSS-RLHF/)

## Chat Models

1. Stanford Alpaca: An Instruction-following LLaMA Model. [[code]](https://github.com/tatsu-lab/stanford_alpaca)
2. Alpaca LoRA. [[code]](https://github.com/tloen/alpaca-lora)
3. Vicuna. [[code]](https://github.com/lm-sys/FastChat)
4. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. [[code]](https://github.com/ZrrSkywalker/LLaMA-Adapter) [[paper]](https://arxiv.org/pdf/2303.16199.pdf) [[v2-paper]](https://arxiv.org/pdf/2304.15010.pdf)
5. Stable Vicuna [[project]](https://stability.ai/blog/stablevicuna-open-source-rlhf-chatbot)
6. Koala: A Dialogue Model for Academic Research. [[paper]](https://bair.berkeley.edu/blog/2023/04/03/koala/) [[code]](https://github.com/young-geng/EasyLM)
7. Open-Assistant. [[project]](https://github.com/LAION-AI/Open-Assistant)
8. GPT4ALL. [[code]](https://github.com/nomic-ai/gpt4all) [[demo]](https://gpt4all.io/index.html)
9.  ChatPLUG: Open-Domain Generative Dialogue System with Internet-Augmented Instruction Tuning for Digital Human. [[paper]](https://arxiv.org/abs/2304.07849) [[code]](https://github.com/X-PLUG/ChatPLUG)
10. CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society. [[paper]](https://arxiv.org/abs/2303.17760) [[code]](https://github.com/camel-ai/camel)
11. MPTChat. [[blog]](https://www.mosaicml.com/blog/mpt-7b) [[code]](https://github.com/mosaicml/llm-foundry/)
12. ChatGLM2 [[code]](https://github.com/THUDM/ChatGLM2-6B)


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

1. CLIP: Learning Transferable Visual Models From Natural Language Supervision. [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP)
2. ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation. [[paper]](https://arxiv.org/abs/2107.07651) [[code]](https://github.com/salesforce/ALBEF)
3. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. [[paper]](https://arxiv.org/abs/2201.12086) [[code]](https://github.com/salesforce/BLIP)
4. mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality [[paper]](https://arxiv.org/abs/2304.14178) [[code]](https://arxiv.org/abs/2304.14178) [[dome]](https://arxiv.org/abs/2304.14178) [[blog]](https://zhuanlan.zhihu.com/p/625631667)
5. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models [[code]](https://link.zhihu.com/?target=https%3A//github.com/salesforce/LAVIS/tree/main/projects/blip2)
6. Kosmos-1: Language Is Not All You Need: Aligning Perception with Language Models. [[paper]](https://arxiv.org/abs/2302.14045) [[code]](https://github.com/microsoft/unilm)
7. Versatile Diffusion: Text, Images and Variations All in One Diffusion Model [[code]](https://link.zhihu.com/?target=https%3A//github.com/SHI-Labs/Versatile-Diffusion)
8. LLaVA: Large Language and Vision Assistant. [[paper]](https://arxiv.org/abs/2304.08485) [[project]](https://llava-vl.github.io/) [[blog]](https://zhuanlan.zhihu.com/p/622907299)
9. PaLM-E: An Embodied Multimodal Language Model. [[paper]](https://arxiv.org/abs/2303.03378) [[code]](https://palm-e.github.io/)
10. BEiT-3: Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. [[paper]](https://arxiv.org/abs/2208.10442) 
11. X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages. [[paper]](https://arxiv.org/abs/2305.04160) 
12. IMAGEBIND: One Embedding Space To Bind Them All. [[paper]](https://arxiv.org/pdf/2305.05665.pdf) [[code]](https://facebookresearch.github.io/ImageBind)
13. PaLM 2. [[paper]](https://ai.google/static/documents/palm2techreport.pdf)
14. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. [[paper]](http://arxiv.org/abs/2305.06500)


## Visual Chat Models

1. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. [[paper]](http://arxiv.org/abs/2304.10592) [[code]](https://minigpt-4.github.io/)
2. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. [[code]](https://github.com/ZrrSkywalker/LLaMA-Adapter) [[paper]](https://arxiv.org/pdf/2303.16199.pdf) [[v2-paper]](https://arxiv.org/pdf/2304.15010.pdf)
3. MMGPT：MultiModal-GPT: A Vision and Language Model for Dialogue with Humans. [[paper]](https://arxiv.org/abs/2305.04790) [[code]](https://github.com/open-mmlab/Multimodal-GPT)
4. InternChat: Solving Vision-Centric Tasks by Interacting with Chatbots Beyond Language [[paper]](https://arxiv.org/pdf/2305.05662.pdf) [[code]](https://github.com/OpenGVLab/InternChat)
5. VideoChat : Chat-Centric Video Understanding. [[paper]](https://arxiv.org/pdf/2305.06355.pdf)
6. Otter: A Multi-Modal Model with In-Context Instruction Tuning. [[paper]](https://arxiv.org/pdf/2305.03726.pdf) [[code]](https://github.com/Luodian/Otter)
7. DetGPT: Detect What You Need via Reasoning.  [[paper]](http://arxiv.org/abs/2306.09224)
8. VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks. [[paper]](https://arxiv.org/abs/2305.11175)
9. LLaVA: Large Language and Vision Assistant. [[paper]](https://arxiv.org/abs/2304.08485) [[project]](https://llava-vl.github.io/) [[blog]](https://zhuanlan.zhihu.com/p/622907299)
10. VisualGLM. [[code]](https://github.com/THUDM/VisualGLM-6B/tree/main)
11. PandaGPT: One Model to Instruction-Follow Them All. [[project]](https://panda-gpt.github.io/)
12. ChatSpot. [[demo]](https://link.zhihu.com/?target=https%3A//chatspot.streamlit.app/)

## Datasets

1. DataComp: In search of the next generation of multimodal datasets. [[paper]](https://arxiv.org/pdf/2304.14108.pdf) [[project]](https://www.datacomp.ai.)


## Evaluation
1. MME. [[paper]](https://arxiv.org/abs/2306.13394)
2. Multimodal Chatbot Areana. [[demo]](http://vlarena.opengvlab.com/)


## 

有一些更有影响力的仓库总结了大模型的相关工作：
- [NLP] [LLM](https://github.com/RUCAIBox/LLMSurvey/tree/main)
- [MM] [MLLM](https://github.com/bradyfu/awesome-multimodal-large-language-models)


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