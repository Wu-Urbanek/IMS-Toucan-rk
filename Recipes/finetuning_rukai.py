import os
import torch
from Modules.ToucanTTS.ToucanTTS import ToucanTTS
from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_rukai
from Utility.corpus_preparation import prepare_tts_corpus
from Utility.storage_config import MODEL_DIR
from Utility.storage_config import PREPROCESSING_DIR

def run(gpu_id, resume_checkpoint, finetune, model_save_dir, cache_dir, local_dataset):
    """
    Rukai TTS Finetuning Recipe
    """
    # 1. 設定路徑 (這些路徑會對應到你的 Modal Volume)
    if model_save_dir is None:
        model_save_dir = "/data/rukai_test/models/rukai_v1"
    if cache_dir is None:
        cache_dir = "/data/rukai_test/cache/rukai_v1"
        
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 2. 準備語料 (這會自動呼叫你寫的 build_path_to_transcript_rukai)
    # 並執行特徵提取與對齊
    print("Preparing Rukai Corpus...")
    train_set, valid_set, device = prepare_tts_corpus(
        path_to_transcript_dict=build_path_to_transcript_rukai(),
        corpus_id="rukai_corpus",
        lang="dru",  # 魯凱語 ISO 639-3 代碼
        save_dir=cache_dir
    )

    # 3. 初始化模型
    # 我們選擇使用 ToucanTTS 架構
    model = ToucanTTS()

    # 4. 啟動訓練迴圈
    # 因為是 25 句的 Sanity Check，我們設定較小的 batch size 並從預訓練模型開始
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=device,
               save_directory=model_save_dir,
               batch_size=8,  # 小規模資料建議 batch 小一點
               eval_lang="dru",
               warmup_steps=500,
               steps=2000,     # 先跑 2000 步看成果
               lr=0.0001,
               resume_checkpoint=resume_checkpoint,
               use_wandb=False, # 先關掉 wandb 以免沒設定 key 報錯
               finetune=finetune)