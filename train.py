import torch
from torch import optim
import torch.nn as nn
import time
import math
from random import shuffle
import os

from data_process import prepare_data, tensorFromSentence
from model import Encoder, Decoder, Translator


SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()


# def indexesFromSentence(lang, sentence):
#     indexes = []
#     for word in sentence.split(' '):
#         if word in lang.word2index.keys():
#             indexes.append(lang.word2index[word])
#         else:
#             indexes.append(lang.word2index['UNK'])
#     return indexes
#
#
# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
#
#
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def batchify(data, input_lang, output_lang, batch_size, shuffle_data=True):
    if shuffle_data:
        shuffle(data)
    number_of_batches = len(data) // batch_size
    batches = list(range(number_of_batches))
    longest_elements = list(range(number_of_batches))

    for batch_number in range(number_of_batches):
        longest_input = 0
        longest_target = 0
        input_variables = list(range(batch_size))
        target_variables = list(range(batch_size))
        index = 0
        for pair in range((batch_number * batch_size), ((batch_number + 1) * batch_size)):
            input_variables[index], target_variables[index] = tensorsFromPair(input_lang, output_lang, data[pair])
            if len(input_variables[index]) >= longest_input:
                longest_input = len(input_variables[index])
            if len(target_variables[index]) >= longest_target:
                longest_target = len(target_variables[index])
            index += 1
        batches[batch_number] = (input_variables, target_variables)
        longest_elements[batch_number] = (longest_input, longest_target)
    return batches, longest_elements, number_of_batches


def pad_batch(batch):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch[0], padding_value=EOS_token)
    padded_targets = torch.nn.utils.rnn.pad_sequence(batch[1], padding_value=EOS_token)
    return padded_inputs, padded_targets


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


if __name__ == "__main__":

    hidden_size = 512
    learning_rate = 0.01
    n_epochs = 10
    print_every = 100
    batch_size = 128
    save_every = 500
    checkpoint_path = None
    output_directory = '/kaggle/working/saved_models/'

    input_lang, output_lang, pairs = prepare_data('eng', 'rus')
    batches, longest_seq, num_batches = batchify(pairs, input_lang, output_lang,
                                           batch_size=batch_size, shuffle_data=True)

    encoder = Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_lang.n_words, dropout=0.1).to(device)
    model = Translator(encoder, decoder, device, input_lang, output_lang)
    print(encoder)
    print(decoder)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    start = time.time()
    iteration = 0

    if checkpoint_path is not None:
        model, optimizer, learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    for epoch in range(n_epochs):
        print_loss = 0
        for iter, batch in enumerate(batches):
            model.zero_grad()
            input_batch, target_batch = pad_batch(batch)
            preds = model(input_batch, target_batch)
            loss = 0
            for i in range(preds.shape[0]):
                loss += criterion(preds[i].float(), target_batch[i].view(-1))
            print_loss += loss/batch_size
            loss.backward()
            optimizer.step()
            if (iter + 1) % print_every == 0:
                print('Time elapsed: {:.2f}\tIteration: {:.2f}\tloss: {:.5f}\tSpeed: {:.2f} iter/s'.format(
                    time.time()-start, iteration+1, print_loss/print_every, (time.time()-start)/(iteration+1)))
                print_loss = 0
            if (iteration+1) % save_every == 0:
                checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration+1, checkpoint_path)
            iteration += 1
