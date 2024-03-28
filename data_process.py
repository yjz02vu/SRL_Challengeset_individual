import json
import nltk
import datasets
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer
from feature import tokenize, lemmatization, named_entity_recognition, sub_tree, capitalization, syntactic_head, PoS_tag, Tag, dep_relations, dep_path, dep_dist_to_head, extract_morph_inform, is_predicate, extract_wordnet_class

task = "srl"
model_checkpoint = "/home/ziggy/Desktop/Ad_ml/final/base_model"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


label_dict = {'C-ARGM-EXT': 0, 'C-ARG4': 1, 'C-ARGM-DIR': 2, 'ARG5': 3, 'C-ARGM-CXN': 4, 'ARGA': 5, 'C-ARG2': 6, 'C-ARGM-GOL': 7, 'C-V': 8, 'ARGM-MNR': 9, 'R-ARGM-TMP': 10, 'ARGM-LOC': 11, 'ARGM-DIR': 12, 'C-ARGM-TMP': 13, 'C-ARG3': 14, 'C-ARGM-COM': 15, 'ARGM-ADV': 16, '_': 17, 'R-ARGM-GOL': 18, 'C-ARGM-ADV': 19, 'R-ARGM-ADV': 20, 'R-ARG1': 21, 'ARGM-CAU': 22, 'C-ARGM-PRR': 23, 'ARG3': 24, 'C-ARG1-DSP': 25, 'R-ARGM-CAU': 26, 'C-ARGM-LOC': 27, 'R-ARG0': 28, 'R-ARG3': 29, 'ARG1': 30, 'R-ARGM-LOC': 31, 'ARGM-GOL': 32, 'ARGM-DIS': 33, 'ARGM-PRD': 34, 'C-ARG1': 35, 'R-ARGM-MNR': 36, 'ARGM-EXT': 37, 'ARG2': 38, 'ARGM-TMP': 39, 'R-ARG2': 40, 'R-ARG4': 41, 'ARG0': 42, 'ARGM-PRR': 43, 'R-ARGM-DIR': 44, 'ARG1-DSP': 45, 'ARGM-CXN': 46, 'ARGM-PRP': 47, 'C-ARG0': 48, 'C-ARGM-PRP': 49, 'R-ARGM-COM': 50, 'ARGM-REC': 51, 'R-ARGM-ADJ': 52, 'C-ARGM-MNR': 53, 'ARGM-NEG': 54, 'ARGM-COM': 55, 'ARGM-ADJ': 56, 'ARGM-MOD': 57, 'ARG4': 58, 'ARGM-LVB': 59}


def tokenize_conll(sentences):
    """
    tokenize sentences for the conll format
    """
    tokenized_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokenized_sentences.append([(token, i+1) for i, token in enumerate(tokens)])
    return tokenized_sentences

def tokenize_(sentences):
    """
    tokenize sentenecs for a list of sentences consisting tokens
    """
    tokenized_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokenized_sentences.append([token for token in tokens])
    return tokenized_sentences


# def format_conll(tokenized_sentences, map_target_l):
#     """
#     map label to target arguments and predicates and create conll format
#     :tokenized_sentences: list of tokenized_sentences
#     :map_target_l: list of target information of specifict test cases such as "negation", including target token, label, predicates
#     """
#     formatted_sentences = []    
#     for i,s in enumerate(tokenized_sentences):
#         formatted_s = ''
#         token, label, pred = t_target[test_name][i]
#         # print(token, label, pred)
#         for t in s:
#             if t[0] == token:
#                 formatted_s += f"{t[1]}\t{t[0]}\t_\t{label}\n"
#             elif t[0] == pred:
#                 formatted_s += f"{t[1]}\t{t[0]}\t{pred}\tV\n"
#             else:
#                 formatted_s += f"{t[1]}\t{t[0]}\t_\t_\n"

#         formatted_s += '\n'  # Add an empty line to separate sentences
#         formatted_sentences.append(formatted_s)
#     return formatted_sentences

def format_conll(test_sentence, t_target, test_name):
    """
    map label to target arguments and predicates and create conll format
    :test_sentence: all instances for the specific test
    :t_target: list of target information of specifict test cases such as "negation",including target token, label, predicates
    :test_name: 
    """
    #tokenize texts
    
    test_sents = test_sentence[test_name]
    tokenized_sentences = tokenize_conll(test_sents)
    
    #create conll
    formatted_sentences = []    
    for i,s in enumerate(tokenized_sentences):
        formatted_s = ''
        token, label, pred = t_target[test_name][i]
        # print(token, label, pred)
        for t in s:
            if t[0] == token:
                formatted_s += f"{t[1]}\t{t[0]}\t_\t{label}\n"
            elif t[0] == pred:
                formatted_s += f"{t[1]}\t{t[0]}\t{pred}\tV\n"
            else:
                formatted_s += f"{t[1]}\t{t[0]}\t_\t_\n"

        formatted_s += '\n'  # Add an empty line to separate sentences
        formatted_sentences.append(formatted_s)
    return formatted_sentences

def preprocess_data(tokenized_sentences,target_infor):
    """
    produce a list of lists where each contain token dictionaries
    """
  
    
    preprocessed_test = []

    for i, s in enumerate(tokenized_sentences):
        sentlist = []  # Create a new sentlist for each sentence
        token, label, pred = target_infor[i]
        for t in s:
            token_dict = {}
            if t[0] == token:
                token_dict = {"ID": t[1], "form": t[0], "V": "_", "arg": label}
            elif t[0] == pred:
                token_dict = {"ID": t[1], "form": t[0], "V": "V", "arg": "_"}
            else:
                token_dict = {"ID": t[1], "form": t[0], "V": "_", "arg": "_"}

            sentlist.append(token_dict)

        preprocessed_test.append(sentlist)

    return preprocessed_test

# #def preprocess_data(test_sentence,t_target,test_name):
#     """
#     produce a list of lists where each contain token dictionaries
#     """
#     test_sents = test_sentence[test_name]
#     tokenized_sentences = tokenize_conll(test_sents)
    
#     preprocessed_test = []

#     for i, s in enumerate(tokenized_sentences):
#         sentlist = []  # Create a new sentlist for each sentence
#         token, label, pred = t_target[test_name][i]
#         for t in s:
#             token_dict = {}
#             if t[0] == token:
#                 token_dict = {"ID": t[1], "form": t[0], "V": "_", "arg": label}
#             elif t[0] == pred:
#                 token_dict = {"ID": t[1], "form": t[0], "V": "V", "arg": "_"}
#             else:
#                 token_dict = {"ID": t[1], "form": t[0], "V": "_", "arg": "_"}

#             sentlist.append(token_dict)

#         preprocessed_test.append(sentlist)

#     return preprocessed_test

def create_word_sentlist(pre_list):
    '''
    Creating list of token lists with the ARG and V infomation.
    '''
    word_sentlist = []
    for sentence in pre_list:
        featdict = {}
        wordlist,args,pred = [],[],[]
        for token in sentence:
            wordlist.append(token['form'])
            if token['V'] == 'V':
                pred.append(token['form'])
            args.append(label_dict[token['arg']])
        featdict['tokens'],featdict['srl_arg_tags'],featdict['pred'] = wordlist,args,pred
        word_sentlist.append(featdict)
    
    return word_sentlist


def tokenize_and_align_labels(examples, label_all_tokens=True):
    '''
    This function solves label alignment after re-tokenization and creates sent+[SEP]+pred+[SEP] structure
    '''
    tokenized_inputs = tokenizer(examples["tokens"],examples['pred'], truncation=True, is_split_into_words=True)
    labels = []
    all_word_ids = []
    for i, label in enumerate(examples[f"{task}_arg_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if (word_idx is None):
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        all_word_ids.append(word_ids)
    
    
    tokenized_inputs['word_ids'] = all_word_ids
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def write_conll_multiple(input_texts, output_file):
    """
    Write the results of NLP functionalities in CoNLL format to a file for multiple input texts.
    :param input_texts: A list of input texts to analyze.
    :param output_file: The file path to write the output in CoNLL format.
    """
    with open(output_file, 'w') as f:
        for input_text in input_texts:
            # Tokenize the input text
            tokens = tokenize(input_text)
            # Perform Named Entity Recognition (NER)
            entities = named_entity_recognition(input_text)
            # Perform Part-of-Speech (POS) Tagging
            pos_tags = PoS_tag(input_text)
            # Perform morph_relation
            morph = extract_morph_inform(input_text)
            # Perform head_relation
            head = dep_path(input_text)
            #predicate
            pred = is_predicate(input_text)

            for i in range(len(tokens)):
                token = tokens[i]
                ner_tag = entities[i].label_ if i < len(entities) else "_"
                pos_tag = pos_tags[i]['pos']
                morph_relation = morph[i]['morph']
                hed_relation = head[i]['head']
                # pred = pred[i]['is_predicate']
                # Write the token, NER tag, POS tag, and dependency relation in CoNLL format
                f.write(f"{i+1}\t{token}\t{morph_relation}\t{ner_tag}\t{pos_tag}\t{hed_relation}\t_\t_\t_\t_\n")
            # Add an empty line between different input texts
            f.write("\n")
            
            
def read_conll_file(file_path):
    """
    Read a CoNLL file and convert it into a list of lists in CoNLL format
    """
    conll_output = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                sentence.append(parts)
            else:
                if sentence:
                    conll_output.append(sentence)
                    sentence = []
    if sentence:
        conll_output.append(sentence)  # Add the last sentence if not empty
    return conll_output

def preprocess_data_model(conll_output):
    """
    Produce a list of lists where each contains token dictionaries
    """
    preprocessed_data = []

    for sentence in conll_output:
        token_dicts = []

        for token in sentence:
            token_dict = {
                "ID": token[0],
                "form": token[1],
                "morph":token[2],
                "ner":token[3],
                "pos":token[4],
                "head":token[5]
                # "V": token[2],
                # "ARG": token[3]
            }
            token_dicts.append(token_dict)

        preprocessed_data.append(token_dicts)

    return preprocessed_data

def extract_feature_and_label(preprocessed_test, t_tokens, t_label):
    """
    This function extract features and label from extracted feature list of dicts.
    It will flattern list of sentences into list of tokens.
    
    """

    label_list = []
    feature_list = [x for xs in preprocessed_test for x in xs]
    
    for m in feature_list:
        if m["form"] in t_tokens:
            ind = t_tokens.index(m["form"])
            # print(ind)
            label_list.append(t_label[ind])
            # print(t_label[ind])
        else:
            label_list.append("_")
    return feature_list, label_list


def classify_data(model, vec, features):  
    features = vec.transform(features)
    predictions = model.predict(features)
    return predictions

def calculate_failure_rate(gold_labels, predicted_labels):
    """
    Calculate the failure rate based on gold labels and predicted labels.

    Args:
    gold_labels (list): List of gold labels.
    predicted_labels (list): List of predicted labels.

    Returns:
    float: Failure rate.
    """
    # Calculate the total number of tokens
    total_tokens = len(gold_labels)

    # Initialize failure count
    failure_count = 0

    # Iterate through each pair of gold and predicted labels
    for gold_label, pred_label in zip(gold_labels, predicted_labels):
        if gold_label != pred_label:
            failure_count += 1

    # Calculate failure rate
    failure_rate = failure_count / total_tokens

    return failure_rate, failure_count