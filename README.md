# GoyangE
NLP text to cat audio model
<img src="https://github.com/user-attachments/assets/925e2cfc-9a3f-402f-9053-384a77233d86" width="600">




Model architecture:

BPE -> Intent classification (DistilBERT) -> Mel spectrogram -> WaveGAN sound generation -> Fine-tuning

Dataset: 
https://huggingface.co/datasets/zeddez/CatMeows Naming convention: Files containing meows are in the dataset.zip archive. They are PCM streams (.wav).

Naming conventions follow the pattern C_NNNNN_BB_SS_OOOOO_RXX, which has to be exploded as follows:

C = emission context (values: B = brushing; F = waiting for food; I: isolation in an unfamiliar environment); NNNNN = cat’s unique ID; BB = breed (values: MC = Maine Coon; EU: European Shorthair); SS = sex (values: FI = female, intact; FN: female, neutered; MI: male, intact; MN: male, neutered); OOOOO = cat owner’s unique ID; R = recording session (values: 1, 2 or 3) XX = vocalization counter (values: 01..99)

Emission context used for intent classification (B = joy, F = anticipation/demanding, I = fear)
