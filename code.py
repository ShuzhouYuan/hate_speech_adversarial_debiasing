from transformers import DistilBertTokenizer, AdamW, DistilBertModel, DistilBertConfig
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run tokenizer
def run_tokenizer(train_csv, test_csv):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 

    def get_max_len(tokenizer, train_csv):
        tweets = train_csv.tweet.values
        max_length = 0
        for t in tweets:
          ids = tokenizer.encode(t)
          max_length = max(len(ids),max_length)
        return max_length if max_length <= 512 else 512 # The max sequence length for BERT is 512

    max_length = get_max_len(tokenizer, train_csv)
    train_tweets, train_labels = train_csv.tweet.values, train_csv.num_label.values
    test_tweets, test_labels = test_csv.tweet.values, test_csv.num_label.values

    def tokenize_for_tweet(tokenizer, tweets, labels):
        input_ids = []
        attention_masks = []

        for t in tweets:
            input_dict = tokenizer.encode_plus(t, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length',return_tensors='pt')
            input_ids.append(input_dict['input_ids'])
            attention_masks.append(input_dict['attention_mask'])
        input_ids = torch.cat(input_ids,dim=0)
        attention_masks = torch.cat(attention_masks,dim=0)
        labels=torch.tensor(labels)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        return dataset
        
    train_dataset = tokenize_for_tweet(tokenizer, train_tweets, train_labels)
    test_dataset = tokenize_for_tweet(tokenizer, test_tweets, test_labels)
    return train_dataset, test_dataset, tokenizer

# Make data loader
batch_size = 20
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)


# Define classifier and adversary
configuration = DistilBertConfig()

class Classifier(nn.Module):
  def __init__(self, num_label):
    super().__init__()
    self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')
    self.linear = nn.Linear(configuration.hidden_size, num_label)

  def forward(self, input_ids, attention_mask): # input_id [batch_size, sentence_length]
    last_hidden_state = self.bert(input_ids, attention_mask)[0] # last_hidden_state [batch_size, sentence_length, hidden_size]
    last_hidden_state_mean = torch.mean(last_hidden_state, dim=1) # last_hidden_state [batch_size, hidden_size]
    output = self.linear(last_hidden_state_mean) # output [batch_size, num_label]
    return last_hidden_state_mean, output

class Adversary(nn.Module):
  def __init__(self, num_protected_label, hidden_size):
    super().__init__()
    self.linear1 = nn.Linear(configuration.hidden_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, num_protected_label)

  def forward(self, clf_last_state):
    output1 = self.relu(self.linear1(clf_last_state))
    output = self.linear2(output1)
    return output


# Functions for adversarial training and evaluation
def joint_training(clf, adv, epochs, clf_optimizer, adv_optimizer, train_dataloader, test_dataloader, alpha):
    for e in range(epochs):
        print('training {} epoch...'.format(e+1))
        start_time = time.time()

        train_loss, total_clf_loss, total_adv_loss = 0, 0, 0

        clf.train(True)
        adv.train(True)
        for input, mask, label in train_dataloader:
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            label = label.to(DEVICE)

            # make protected labels
            protected_label = torch.tensor([1 if l == 1 else 0 for l in label], dtype=torch.long).to(DEVICE) # 0 no-offensive 1 offensive
            
            clf.zero_grad()
            adv.zero_grad()

            last_hidden_state_mean, clf_output = clf(input_ids=input, attention_mask=mask)

            adv_output = adv(last_hidden_state)

            clf_loss = loss_function(clf_output, label)
            adv_loss = loss_function(adv_output, protected_label)

            # calculate the total loss
            total_loss = clf_loss + alpha*adv_loss

            train_loss += total_loss.item()
            total_clf_loss += clf_loss.item()
            total_adv_loss += adv_loss.item()

            total_loss.backward(retain_graph=True)
            clf_optimizer.step()

            adv_loss.backward()
            adv_optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)
        sec = time.time()-start_time
        print('{} seconds used......'.format(sec))
        print("{} training finished! average train loss: {}".format(e+1,avg_train_loss))
        print('total clf loss: {} total adv loss: {}'.format(total_clf_loss, total_adv_loss))
        print('evaluating...')
        evaluate(clf, test_dataloader)

def evaluate(clf, test_dataloader):
    num_total, num_correct = 0, 0
    clf.train(False)
    with torch.no_grad():
      eval_loss = 0
      true_labels, predict_labels = [], []
      for input, mask, label in test_dataloader:
          clf.zero_grad()
          
          input = input.to(DEVICE)
          mask = mask.to(DEVICE)
          label = label.to(DEVICE)

          last_hidden_state_mean, output = clf(input_ids=input, attention_mask=mask)

          loss = loss_function(output, label)

          predict_label = torch.argmax(output, dim=1)

          true_labels += label.tolist()
          predict_labels += predict_label.tolist()

          num_correct += (predict_label == label).sum().item()
          num_total += len(label)

          eval_loss += loss.item()

      avg_eval_loss = eval_loss / len(test_dataloader)

      acc = num_correct/num_total
    print('average eval_loss: {}, accuracy: {}'.format(avg_eval_loss,acc))

if __name__ == '__main__':
    # Run adversarial training
    classifier = Classifier(3).to(DEVICE)
    adversary = Adversary(2, 200).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    clf_optimizer = AdamW(classifier.parameters(),lr = 2e-5, eps = 1e-8)
    adv_optimizer = torch.optim.AdamW(adversary.parameters(), lr=0.001)

    epochs = 5
    alpha = 2

    joint_training(classifier, adversary, epochs, clf_optimizer, adv_optimizer, train_dataloader, test_dataloader, alpha)


