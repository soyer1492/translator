import re
import torch

SOS_token = 0
EOS_token = 1


ENG_DATA = '/home/soyer1492/dataset/corpus.en_ru.1m.en'
RUS_DATA = '/home/soyer1492/dataset/corpus.en_ru.1m.ru'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 30


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index.keys():
            indexes.append(lang.word2index[word])
        else:
            indexes.append(lang.word2index['UNK'])
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


def normalize_text(text, lang):
    # text = re.sub("([.!?])", " \1", text)
    text = text.lower()
    if lang == 'rus':
        if len(re.sub("[^a-zA-Z0-9]", "", text)) > 0:
            return None
        text = re.sub("[^а-яёА-ЯЁ.!?,]+", " ", text)
    else:
        text = re.sub("[^a-zA-Z.!?,]+", " ", text)
    text = text.replace('.', ' . ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')
    text = text.replace(',', ' , ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    if text.endswith(' '):
        text = text[:-1]
    if len(text.split(' ')) > MAX_LENGTH-1:
        return None
    return text


def prepare_data(lang_from, lang_to):
    text_pairs = []
    with open(ENG_DATA) as f:
        data_en = f.readlines()
    with open(RUS_DATA) as f:
        data_ru = f.readlines()
    for i in range(len(data_en)):
        text_en = normalize_text(data_en[i], 'eng')
        text_ru = normalize_text(data_ru[i], 'rus')
        if text_en and text_ru:
            text_pairs.append([text_en, text_ru])

    in_lang = Language(lang_from)
    out_lang = Language(lang_to)

    for pair in text_pairs:
        in_lang.add_sentence(pair[0])
        out_lang.add_sentence(pair[1])

    in_lang.update_vocab()
    out_lang.update_vocab()

    return in_lang, out_lang, text_pairs


class Language:
    def __init__(self, lang):
        self.lang = lang
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3
        self.min_occur = 20

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
                self.word2count[word] = 1
            else:
                self.word2count[word] += 1

    def update_vocab(self):
        words = ['SOS', 'EOS', 'UNK']
        words.extend([item for item, count in self.word2count.items() if count >= self.min_occur])
        self.word2index = dict([(w, i) for i, w in enumerate(words)])
        self.index2word = dict([(num, word) for word, num in self.word2index.items()])
        self.n_words = len(self.word2index)
