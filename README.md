# GoyangE
NLP text to cat audio model
![Uploading image.png…]()


Model architecture:

BPE -> Sentence Encoding (MiniLM) -> Intent classification (DistilBERT) -> Phonetic Alignment (Wav2Vec) -> Mel spectrogram -> GAN sound generation -> Fine-tuning
