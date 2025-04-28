# GoyangE
NLP text to audio model

Model architecture:

BPE -> Sentence Encoding (MiniLM) -> Intent classification (DistilBERT) -> Phonetic Alignment (Wav2Vec) -> Mel spectrogram -> GAN sound generation -> Fine-tuning
