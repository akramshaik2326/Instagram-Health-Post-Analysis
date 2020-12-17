import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, glob, os.path
from collections import defaultdict 
import torch
from model import EncoderCNN, DecoderRNN

os.chdir('C:/Users/sabdu/')

from data_loader import get_loader

transform_test = transforms.Compose([ 
    transforms.Resize(256),                          
    transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])

# Create the data loader.
data_loader = get_loader(transform=transform_test, mode='test',ipath = '12.jpg')
orig_image, image = next(iter(data_loader))
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

embed_size = 256
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('./coco/models', encoder_file),map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(os.path.join('./coco/models', decoder_file),map_location=torch.device('cpu')))

encoder.to(device)
decoder.to(device)

image = image.to(device)

features = encoder(image).unsqueeze(1)

output = decoder.sample(features)
print('example output:', output)

assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

def clean_sentence(output):
    sentence = ''
    new_output = output[1:-1] 
    for i in range(len(new_output)):
        curr_token = new_output[i]
        if curr_token == 18: sentence += data_loader.dataset.vocab.idx2word[curr_token]
        else: sentence += ' ' + data_loader.dataset.vocab.idx2word[curr_token]
    return sentence

sentence = clean_sentence(output)
print('example sentence:', sentence)

assert type(sentence)==str, 'Sentence needs to be a Python string!'

def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    print(sentence)

get_prediction()

maindir = 'C:/Users/sabdu/Desktop/Test/data/'
filelist = glob.glob(os.path.join(maindir, '*.jpg'))

mydic = {}

for i in range(len(filelist)):
    data_loader = get_loader(transform=transform_test, mode='test',ipath = str(filelist[i].split('\\')[1]))
    orig_image, image = next(iter(data_loader))
    #plt.imshow(np.squeeze(orig_image))
    #plt.title('Image')
    #plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    mydic[str(filelist[i].split('\\')[1])] = sentence
    
import csv    
    
with open('coco.csv', 'w',encoding="utf-8") as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in mydic.items():
        writer.writerow([key, value])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(mydic.values())

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")


Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)

my_file = open('capt10.txt', 'w')
my_file.write('image,cluster,')

for key in mydic:
    Y = vectorizer.transform([mydic[key]])
    pred = model.predict(Y)
    my_file.write(key+','+str(pred[0])+'\\n')

my_file.close()