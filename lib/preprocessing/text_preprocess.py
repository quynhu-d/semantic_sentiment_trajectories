from natasha import (
    Segmenter, MorphVocab,
    NewsNERTagger,
    NewsEmbedding,
    NewsMorphTagger,    
    Doc
)
import re
import glob
import gensim
import en_core_web_lg

from tqdm.auto import tqdm, trange


def prepare_russian_text(input_file, output_file):
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)
    morph_vocab = MorphVocab()

    label_dict = {'NUM': 'ordinal1', 'PRON': 'pron1', 'PER': 'person1'}
    next_label_num = 5
    with open(input_file, encoding='utf-8') as fin:
      raw_text = ' '.join(fin.readlines()).replace('\n', ' ')
    raw_text = re.sub(r'\d+', '0' , raw_text)
    raw_text = ' '.join(re.findall(r'[А-яЁё]+', raw_text))  # remove non-cyrillic symbols

    doc = Doc(raw_text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    for span in reversed(doc.ner.spans):
        if span.type not in label_dict:
            label_dict[span.type] = str(next_label_num)
            next_label_num += 1
        raw_text = "".join((raw_text[:span.start], label_dict[span.type], raw_text[span.stop:]))

    doc = Doc(raw_text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
      
    prepared_text = ''
    prev_num = False
    for token in tqdm(doc.tokens, desc='Text preprocessing...', leave=False):
      
        if token.pos == 'NUM' and not token.text.isdigit():
            if not prev_num:
                prepared_text += '0'
                prepared_text += ' '
                prev_num = True
            continue

        prev_num = False

        if token.pos in label_dict:
            prepared_text += label_dict[token.pos]
            prepared_text += ' '
            
        elif token.pos != 'PUNCT':
                try:
                    token.lemmatize(morph_vocab)
                    prepared_text += token.lemma.lower()
                    prepared_text += ' '
                except Exception as ex:
                    prepared_text += token.text.lower()
                    prepared_text += ' '
    with open(output_file, 'w', encoding='utf-8') as fout:
       fout.write(prepared_text)


#This code is for decontracting shortened words (won't -> will not)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won['’‘`]t", "will not", phrase)
    phrase = re.sub(r"can['’‘`]t", "can not", phrase)
    phrase = re.sub(r"ain['’‘`]t", "am not", phrase)

    # general
    phrase = re.sub(r"n['’‘`]t", " not", phrase)
    phrase = re.sub(r"['’‘`]re", " are", phrase)
    phrase = re.sub(r"['’‘`]s", " is", phrase)
    phrase = re.sub(r"['’‘`]d", " would", phrase)
    phrase = re.sub(r"['’‘`]ll", " will", phrase)
    phrase = re.sub(r"['’‘`]t", " not", phrase)
    phrase = re.sub(r"['’‘`]ve", " have", phrase)
    phrase = re.sub(r"['’‘`]m", " am", phrase)

    #phrase = re.sub('([.;!?])', r' \1 ', phrase)
    phrase = re.sub(r'[^\w.?!;]', ' ', phrase)
    phrase = re.sub(' +', ' ', phrase)
    sentences = re.split('([.;!?] *)', phrase)

    return ' '.join([i.capitalize() for i in  sentences])

def prepare_english_text(input_file, output_file):
    '''
    input_file - path to the txt file containing raw text in English
    output_file - path to txt file, where the preprocessed text will be written 
    '''
    nlp = en_core_web_lg.load(disable=['parser'])
    nlp.max_length = 5000000
    #We will replace PROPN, PRON and NUM with tokens
    #(proper nouns, pronouns and numericals)
    pos_dict = {'PROPN': 'person1', 'PRON': 'pron1', 'NUM': 'ordinal1'}   
    fin  = open(input_file, 'r', encoding='utf-8')
    with open(output_file, 'w', encoding='utf-8') as prepared_text:
      for line in fin:
        #decontract text
        preprocessed_text = decontracted(line.strip())
        #apply nlp model to get lemmas and pos
        nlp_doc = nlp(preprocessed_text)
        for token in nlp_doc:
              #if the pos of the word is PROPN, PRON or NUM
              #we replace it with the token
              if token.pos_ in pos_dict:
                    prepared_text.write(pos_dict[token.pos_])
                    prepared_text.write(' ')
              #if the word is a number, we replace it with token
              elif token.lemma_.isdigit():
                    prepared_text.write('ordinal1')
                    prepared_text.write(' ')
              #we skip punktuation
              elif token.pos_ != 'PUNCT':
                    #we replace word with its lemma
                    prepared_text.write(token.lemma_.lower())
                    prepared_text.write(' ')
        prepared_text.write('\n')