import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda

from data_process import normalize_text, tensorFromSentence
# from train import tensorFromSentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2

MAX_LENGTH = 30


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers=2, dropout=0.1, bidirectional=True):
        super(Encoder, self).__init__()
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=False)
        self.fc = nn.Linear(hidden_size * self.directions, hidden_size)

    def forward(self, input_data, h_hidden, c_hidden):
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        embedded_data = embedded_data.view(input_data.shape[0], input_data.shape[1], -1)
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))

        return hiddens, outputs

    def create_init_hiddens(self, batch_size):
        h_hidden = Variable(torch.zeros(self.num_layers * self.directions,
                                        batch_size, self.hidden_size))
        c_hidden = Variable(torch.zeros(self.num_layers * self.directions,
                                        batch_size, self.hidden_size))
        if torch.cuda.is_available():
            return h_hidden.cuda(), c_hidden.cuda()
        else:
            return h_hidden, c_hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, layers=2, dropout=0.1, bidirectional=True):
        super(Decoder, self).__init__()

        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_learner = nn.Linear(hidden_size * self.directions,
                                       hidden_size * self.directions)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=False)
        self.context_combiner = nn.Linear((hidden_size * self.directions)
                                          + (hidden_size * self.directions), hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        batch_size = embedded_data.shape[1]
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        top_hidden = outputs[0].view(self.num_layers, self.directions,
                                     hiddens.shape[1],
                                     self.hidden_size)[self.num_layers - 1]
        top_hidden = top_hidden.permute(1, 2, 0).contiguous().view(batch_size, -1, 1)

        prep_scores = self.score_learner(encoder_hiddens.permute(1, 0, 2))
        scores = torch.bmm(prep_scores, top_hidden)
        attn_scores = self.soft(scores)
        con_mat = torch.bmm(encoder_hiddens.permute(1, 2, 0), attn_scores)
        h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat,
                                                             top_hidden), dim=1).view(batch_size, -1)))
        pred = self.output(h_tilde)
        pred = self.log_soft(pred)
        return pred, outputs


class Translator(nn.Module):
    def __init__(self, encoder, decoder, device, lang_from, lang_to):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.lang_to = lang_to
        self.lang_from = lang_from

    def forward(self, input_batch, target_batch):
        batch_size = input_batch.shape[1]

        enc_h_hidden, enc_c_hidden = self.encoder.create_init_hiddens(batch_size)

        enc_hiddens, enc_outputs = self.encoder(input_batch, enc_h_hidden, enc_c_hidden)

        outputs = []

        decoder_input = torch.Tensor(1, input_batch.shape[1]).fill_(SOS_token).to(self.device)
        decoder_input = decoder_input.long()

        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]

        for i in range(target_batch.shape[0]):
            pred, dec_outputs = self.decoder(decoder_input, dec_h_hidden,
                                             dec_c_hidden, enc_hiddens)
            outputs.append(pred)
            decoder_input = target_batch[i].view(1, -1)
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]

        outputs = torch.stack(outputs).to(device)
        return outputs

    def inference(self, sentence):
        sentence = normalize_text(sentence, self.lang_from.lang)
        # words = sentence.split(' ')
        sent2tensor = tensorFromSentence(self.lang_from, sentence)
        input_batch = sent2tensor.view(sent2tensor.shape[0],1,1)
        batch_size = input_batch.shape[1]

        enc_h_hidden, enc_c_hidden = self.encoder.create_init_hiddens(batch_size)

        enc_hiddens, enc_outputs = self.encoder(input_batch, enc_h_hidden, enc_c_hidden)

        outputs = []

        decoder_input = torch.Tensor(1, input_batch.shape[1]).fill_(SOS_token).to(self.device)
        decoder_input = decoder_input.long()

        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]

        # for i in range(input_batch.shape[0]):
        while True:
            pred, dec_outputs = self.decoder(decoder_input, dec_h_hidden,
                                             dec_c_hidden, enc_hiddens)
            pred = pred.argmax(dim=1).view(1, -1).long()
            if pred[0].item() == EOS_token:
                break
            outputs.append(pred[0].item())
            decoder_input = pred
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]

        # outputs = torch.stack(outputs)
        return outputs
