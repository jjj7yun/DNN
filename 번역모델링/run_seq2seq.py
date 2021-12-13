import torch
import argparse
import nltk.translate.bleu_score as bleu

from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import BucketIterator, Field

from time import time
from tqdm import tqdm

from utils import *

def main(args):
    MODEL_NAME = 'Transformer_seq2seq'
    HIDDEN_DIM = args.hidden_dim
    NUM_ENC_LAYERS = args.num_enc_layers
    NUM_DEC_LAYERS = args.num_dec_layers
    NUM_ENC_HEADS = args.num_enc_heads
    NUM_DEC_HEADS = args.num_dec_heads
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    RANDOM_STATE = args.random_state
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("... RUNNING ...")
    print(f"> MODEL_NAME: {MODEL_NAME}")
    print(f"> HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"> NUM_ENC_LAYERS: {NUM_ENC_LAYERS}")
    print(f"> NUM_DEC_LAYERS: {NUM_DEC_LAYERS}")
    print(f"> NUM_ENC_HEADS: {NUM_ENC_HEADS}")
    print(f"> NUM_DEC_HEADS: {NUM_DEC_HEADS}")
    print(f"> NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"> BATCH_SIZE: {BATCH_SIZE}")
    print(f"> LEARNING_RATE: {LEARNING_RATE}")
    print(f"> RANDOM_STATE: {RANDOM_STATE}\n")

    torch.random.manual_seed(RANDOM_STATE)

    # PREPARE DATA FOR TRANSLATION
    SRC = Field(tokenize=str.split, batch_first=True, init_token='<sos>', eos_token='<eos>', fix_length=100)
    TRG = Field(tokenize='spacy', tokenizer_language='en', lower=True, batch_first=True, init_token='<sos>', eos_token='<eos>', fix_length=100)

    train_data = TranslationDataset(path='data/kor-eng_small/kor-eng.train', 
                                    exts=('.ko', '.en'),
                                    fields=([SRC, TRG]))
                                
    valid_data = TranslationDataset(path='data/kor-eng_small/kor-eng.valid',
                                    exts=('.ko', '.en'),
                                    fields=([SRC, TRG]))

    test_data = TranslationDataset(path='data/kor-eng_small/kor-eng.test', 
                                   exts=('.ko', '.en'),
                                   fields=(SRC, TRG))

    # BUILD VOCAB
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    MODEL_CONFIG=(DEVICE, HIDDEN_DIM, NUM_ENC_LAYERS, NUM_DEC_LAYERS, NUM_ENC_HEADS, NUM_DEC_HEADS, SRC, TRG)

    # LOAD DATA
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                           batch_size=BATCH_SIZE,
                                                                           device=DEVICE)

    # DEFINE MODEL
    import model
    MODEL_CLASS = getattr(model, MODEL_NAME)
    model = MODEL_CLASS(*MODEL_CONFIG)

    # Initialize model
    model.apply(initialize_weights)

    # START TRAINING
    print("... TRAINING ...\n")
    s = time()
    model.train_model(NUM_EPOCHS, LEARNING_RATE, train_iterator, valid_iterator)

    print(f"\n... DONE ... TOTAL TRAINING TIME: {time()-s:.2f}s\n")

    # START TEST
    with torch.no_grad():
        bleu_score = 0.0
        test_loss = model.predict(test_iterator)
        for idx in tqdm(range(len(test_data.examples)), desc="Translating test data", dynamic_ncols=True):
            src = vars(test_data.examples[idx])['src']
            trg = vars(test_data.examples[idx])['trg']
            translation = model.translate_sentence(src)

            bleu_score += bleu.sentence_bleu([trg], translation, weights=(1, 0, 0, 0))
            if idx < 5:
                print("Source: ", ' '.join(src))
                print("Target: ",' '.join(trg))
                print("Predict: ",' '.join(translation))
                print()

        bleu_score = bleu_score/len(test_data.examples)
        print(f'\n>>>> BLEU score: {bleu_score:.3f} / Test loss: {test_loss:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden vector')
    parser.add_argument('--num_enc_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num_dec_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--num_enc_heads', type=int, default=8, help='Number of encoder heads')
    parser.add_argument('--num_dec_heads', type=int, default=8, help='Number of decoder heads')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help="Number of data per a batch")
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--random_state', type=float, default=506)

    args = parser.parse_args()

    main(args)