!git clone https://github.com/Beckendrof/intent-classification/

# %%
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from sklearn import preprocessing
import numpy
import torch

le = preprocessing.LabelEncoder()
save_directory = "models/"

loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)
le.classes_ = numpy.load('models/classes.npy', allow_pickle=True)

sample = input("Input Sentence: ")
predict_input = loaded_tokenizer.encode(sample,
                                truncation=True,
                                padding=True,
                                return_tensors="tf")

output = loaded_model(predict_input)
output_n = output[0].numpy()
final = torch.from_numpy(output_n)

# %%
prediction_value = torch.argmax(final, axis=1).numpy()[0]
print("predicted value : ", le.inverse_transform([prediction_value])[0])


# %%
