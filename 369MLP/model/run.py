import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import load_data, multi_acc

def train_model(num_features, model, optimizer, criterion, data_loader, num_epochs, valid_epoch):    
    train_loader, valid_loader = data_loader

    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        epoch_pred = []
        train_y = []

        for batch_data in train_loader:
            batch_x, batch_y = batch_data[:, :num_features], batch_data[:, num_features]
            
            # ----------------------------------- #
            # FILL THIS PART TO COMPLETE THE CODE #
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss=criterion(pred_y,batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            epoch_pred.append(pred_y.detach().numpy())
            train_y.append(batch_y)
            #                                     #
            # ----------------------------------- #

        if epoch % valid_epoch == 0:
            valid_pred = []
            valid_y = []
            with torch.no_grad():

                for batch_data in valid_loader:
                    batch_x, batch_y = batch_data[:, :num_features], batch_data[:, num_features]
                    pred_y = model(batch_x)

                    valid_pred.append(pred_y)
                    valid_y.append(batch_y)

            print(f'>> {epoch} epoch  epoch_loss: {epoch_loss:.4f} train_acc: {multi_acc(epoch_pred, train_y)}% valid_acc: {multi_acc(valid_pred, valid_y)}%')

def main(args):
    torch.random.manual_seed(506)

    GAME_NAME = args.game_name
    assert GAME_NAME in ["369", "u369"], "Game_name must be either '369' or 'u369'"

    if GAME_NAME == "369": 
        NUM_CLASSES = 2
    elif GAME_NAME == "u369":
        NUM_CLASSES = 4
            
    MODEL_NAME = args.model_name
    assert MODEL_NAME in ["categorical", "numerical"], "MODEL_NAME must be either 'categorical' or 'numerical'"

    if MODEL_NAME == "numerical":
        NUM_FEATURES = 1
        MODEL_CONFIG = (NUM_FEATURES, NUM_CLASSES)
    elif MODEL_NAME == "categorical":
        NUM_CATEGORIES = 10 ## NUMBER OF DIGITS
        NUM_FEATURES = 5
        MODEL_CONFIG = (NUM_CATEGORIES, NUM_FEATURES, NUM_CLASSES)
    
    import model
    MODEL_CLASS = getattr(model, MODEL_NAME)

    NUM_TRAIN_DATA = args.num_train_data
    TEST_DATA_OOD = args.test_data_ood
    NUM_EPOCHS = args.num_epochs
    VALID_EPOCH = args.valid_epoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    assert NUM_TRAIN_DATA <= 10000 and NUM_TRAIN_DATA > 0, "Number of train data must be between 1 and 10000"

    print("... RUNNING ...")
    print(f"> GAME_NAME: {GAME_NAME}")
    print(f"> NUM_CLASSES: {NUM_CLASSES}")
    print(f"> MODEL_NAME: {MODEL_NAME}")
    print(f"> NUM_TRAIN_DATA: {NUM_TRAIN_DATA}")
    print(f"> TEST_DATA_OOD: {TEST_DATA_OOD}")
    print(f"> NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"> VALID_EPOCH: {VALID_EPOCH}")
    print(f"> BATCH_SIZE: {BATCH_SIZE}")
    print(f"> LEARNING_RATE: {LEARNING_RATE}\n")

    # PREPARE DATA
    train_data = open(f'data/train_data_{GAME_NAME}.tsv')
    valid_data = open(f'data/valid_data_{GAME_NAME}.tsv')
    if TEST_DATA_OOD:
        test_data = open(f'data/test_data_{GAME_NAME}_ood.tsv')
    else:
        test_data = open(f'data/test_data_{GAME_NAME}.tsv')

    train_data_array = load_data(train_data, NUM_FEATURES, NUM_TRAIN_DATA)
    train_loader = DataLoader(train_data_array, batch_size=BATCH_SIZE)

    valid_data_array = load_data(valid_data, NUM_FEATURES)
    valid_loader = DataLoader(valid_data_array, batch_size=BATCH_SIZE)

    test_data_array = load_data(test_data, NUM_FEATURES)
    test_loader = DataLoader(test_data_array, batch_size=BATCH_SIZE)
    
    data_loader = (train_loader, valid_loader)

    # DEFINE MODEL
    model = MODEL_CLASS(*MODEL_CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # START TRAINING
    print("... TRAINING ...")
    train_model(NUM_FEATURES, model, optimizer, criterion, data_loader, NUM_EPOCHS, VALID_EPOCH)
    
    # START TEST
    with torch.no_grad():
        test_pred = []
        test_y = []
        for b, batch_data in enumerate(test_loader):
            batch_x, batch_y = batch_data[:, :NUM_FEATURES], batch_data[:, NUM_FEATURES]
            pred_y = model(batch_x)

            test_pred.append(pred_y)
            test_y.append(batch_y)
                    
        print(f'\n>>>> TEST_DATA_OOD: {TEST_DATA_OOD} / test accuracy: {multi_acc(test_pred, test_y)}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', type=str, default="369", help="369 or u369")
    parser.add_argument('--model_name', type=str, default="categorical", help="categorical or numerical")

    parser.add_argument('--test_data_ood', action='store_true', help='Use out-of-distribution test set' )
    parser.add_argument('--num_train_data', type=int, default=100, help='Number of train data' )

    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--valid_epoch', type=int, default=1, help='Validate model for every n epochs')
    parser.add_argument('--batch_size', type=int, default=8, help="Number of data per a batch")
    parser.add_argument('--learning_rate', type=float, default=1)

    args = parser.parse_args()

    main(args)