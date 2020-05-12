import re
import nltk
import random
import numpy as np
# nltk.download('abc')
# nltk.download('punkt')
from nltk.corpus import abc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D

class ABCdataset(data.Dataset):
    def __init__(self, words):
        super(ABCdataset, self).__init__()
        self.window_size = 5
        self.all_words = words

    def __getitem__(self, index):
        curr_word = self.all_words[index]
        context_limit = np.random.randint(2, self.window_size+1)

        if index < self.window_size:
        	start, stop = 0, self.window_size + 1
        elif index > len(self.all_words) - self.window_size:
        	start, stop = len(self.all_words) - self.window_size - 2, len(self.all_words)-1 
        else:
        	start, stop = index - context_limit, index + context_limit
        
        label = list(self.all_words[start:index]+self.all_words[index+1:stop]) 
        datap = [curr_word]*len(label)
        return datap, label

    def __len__(self):
      return len(self.all_words)

def init_data():
	all_sentence_data = abc.sents()
	all_words = []
	for sent in all_sentence_data:
		for word in sent:
			if word not in ["!",",","?",'"',"(",")",".",":",";"]:
				all_words.append(word.lower())
	return all_words

class SkipGram(nn.Module):
	def __init__(self, n_vocab, freqs):
		super().__init__()

		self.n_vocab = n_vocab

		freqs = sorted(freqs.values(), reverse=True)
		word_freqs = np.array(freqs)
		unigram_dist = word_freqs/word_freqs.sum()
		noise_d = unigram_dist**(0.75)/np.sum(unigram_dist**(0.75))
		
		self.noise_dist = torch.from_numpy(noise_d)
		self.in_embed = nn.Embedding(n_vocab,300)
		self.in_embed.weight.data.uniform_(-1,1)

		self.out_embed = nn.Embedding(n_vocab,300)
		self.out_embed.weight.data.uniform_(-1,1)
        
	def forward(self, input_words, output_words, batch_size):
		embed_size = 300
		input_vector = self.in_embed(input_words)
		output_vector = self.out_embed(output_words)
		noise_words = torch.multinomial(self.noise_dist, batch_size*5, replacement=True)

		#(batch_size, n_samples, n_embed)
		noise_vector = self.out_embed(noise_words).view(batch_size, 5, embed_size) 
		input_vectors = input_vector.view(batch_size, embed_size, 1)
		output_vectors = output_vector.view(batch_size, 1, embed_size)

		activation = nn.LogSigmoid()
		out_loss = activation(torch.bmm(output_vectors, input_vectors))
		out_loss = out_loss.squeeze()

		noise_loss = activation(torch.bmm(noise_vector*-1, input_vectors))
		noise_loss = noise_loss.squeeze().sum(1)

		return out_loss, noise_loss

def vis_embeddings(embeddings, int_to_vocab, vocab_to_int, epoch):

	print("Creating Visual Embeddings...")
	words_wp = []
	embeddings_wp = []
	for word, ind in vocab_to_int.items():
		words_wp.append(word)
		vector = embeddings(torch.LongTensor(np.array([ind])))
		embeddings_wp.append(vector[0].detach().numpy())

	tsne_wp_3d = TSNE(perplexity=20, n_components=3, init='pca', n_iter=2000, random_state=12)
	embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp[:2500])
	fig = plt.figure()
	ax = Axes3D(fig)
	colors = cm.rainbow(np.linspace(0, 1, 1))
	label="First 2500 words of ABC corpus"
	plt.scatter(embeddings_wp_3d[:, 0], embeddings_wp_3d[:, 1], embeddings_wp_3d[:, 2], c=colors, alpha=1, label=label)
	plt.legend(loc=4)
	plt.title("Epoch "+str(epoch+1)+": Visual Embeddings")
	plt.savefig("Epoch"+str(epoch+1)+"_vis.jpg")

def train(train_words, int_to_vocab, vocab_to_int, freqs):
	print_every = 100
	epochs = 12
	loss_vals = []

	model = SkipGram(len(vocab_to_int), freqs)
	optimizer = optim.Adam(model.parameters(), lr=0.003)

	savepath = "models/"
	train_set = ABCdataset(train_words)
	batch_size = 251
	index_limit = len(train_set) - len(train_set)%batch_size

	print("Training starts...")
	for ep in range(epochs):
	    steps = 0
	    for batchid in range(0,index_limit,batch_size):
	        batch_x, batch_y = [], []
	        for datap in range(batch_size):
	        	xx, yy = train_set[batchid+datap]
	        	batch_x.extend(xx)
	        	batch_y.extend(yy)

	        steps += 1
	        inputs = Variable(torch.LongTensor(batch_x))
	        targets = Variable(torch.LongTensor(batch_y))
	        out_loss, noise_loss = model(inputs, targets, inputs.shape[0])
	        loss = -1*torch.mean(out_loss + noise_loss)

	        if steps % print_every == 0:
	            extent = index_limit//batch_size
	            print("Epoch: {}/{} | Steps: {}/{} | Loss: {}".format(ep+1, epochs,steps,extent,loss.item()))
	    
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	        loss_vals.append(loss.item())

	    torch.save(model.state_dict(), savepath+'model_epoch'+str(ep+1)+'.pth')
	    embeddings = model.in_embed
	    vis_embeddings(embeddings, int_to_vocab, vocab_to_int, ep)

	loss_vals = np.array(loss_vals)
	# plt.plot(loss_vals)
	# plt.savefig('loss.png')

if __name__ == "__main__":

	all_words = init_data()
	word_counts = Counter(all_words)

	final_words = []
	for word in all_words:
		if word_counts[word]>5:
			final_words.append(word)

	word_counts = Counter(final_words)
	sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
	int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
	vocab_to_int = dict(map(reversed, int_to_vocab.items())) 

	int_words = [vocab_to_int[word] for word in final_words]
	word_counts = Counter(int_words)

	freqs = {word: count/len(int_words) for word, count in word_counts.items()}
	p_drop = {word: np.sqrt(1e-5/freqs[word]) for word in word_counts}

	train_words = []
	for word in int_words:
		if random.random() < p_drop[word]:
			train_words.append(word)

	print("Total words: {}".format(len(train_words)))
	print("Unique words: {}".format(len(set(train_words))))

	train(train_words, int_to_vocab, vocab_to_int, freqs)