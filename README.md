# GoyangE
NLP text to cat audio model
![image](https://github.com/user-attachments/assets/925e2cfc-9a3f-402f-9053-384a77233d86)



Model architecture:

BPE -> Sentence Encoding (MiniLM) -> Intent classification (DistilBERT) -> Mel spectrogram -> GAN sound generation -> Fine-tuning
