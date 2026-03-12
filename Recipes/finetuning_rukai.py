import os
from huggingface_hub import hf_hub_download
import torch
from Modules.ToucanTTS.ToucanTTS import ToucanTTS
from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_rukai
from Utility.corpus_preparation import prepare_tts_corpus


def run(
    gpu_id="cpu",
    resume_checkpoint=None,
    resume=False,
    finetune=False,
    model_dir=None,
    use_wandb=False,
    wandb_resume_id=None,
    gpu_count=1,
    **kwargs
):
    """
    Rukai TTS Finetuning Recipe
    """

    # 1) save dir: 對接 run_training_pipeline.py 的 --model_save_dir -> model_dir
    if model_dir is None:
        model_dir = "/data/rukai_test/models/rukai_v1"
    os.makedirs(model_dir, exist_ok=True)

    # 2) cache / preprocessing dir（你自己定義）
    cache_dir = "/data/rukai_test/cache/rukai_v1"
    os.makedirs(cache_dir, exist_ok=True)

    # 3) 準備語料
    print("Preparing Rukai Corpus...")
    dataset = prepare_tts_corpus(
        transcript_dict=build_path_to_transcript_rukai(),
        corpus_dir=cache_dir,
        lang="dru",
        fine_tune_aligner=False,  # sanity check 先關
        gpu_count=gpu_count,
        rank=0,
    )
    # train_set, valid_set, device = prepare_tts_corpus(
    #     transcript_dict = build_path_to_transcript_rukai(),
    #     corpus_dir=cache_dir,
    #     lang="dru",
    #     fine_tune_aligner=False,   # sanity check 先關
    #     gpu_count=gpu_count,
    #     rank=0
    # )
    # for sanity check
    train_set = dataset
    valid_set = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) checkpoint 決策
    if resume_checkpoint is None and not resume:
        resume_checkpoint = hf_hub_download(
            repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt"
        )

    model = ToucanTTS()

    train_loop(
        net=model,
        datasets=[train_set],
        train_samplers=[torch.utils.data.RandomSampler(train_set)],
        gpu_count=1,
        device=device,
        save_directory=model_dir,
        path_to_checkpoint=resume_checkpoint,
        lr=1e-5,
        resume=True,
        warmup_steps=500,
        use_wandb=use_wandb,
        batch_size=12,
        eval_lang="dru",
        fine_tune=True,
        steps=5000,
    )
