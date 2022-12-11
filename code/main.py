import os, sys, getopt, time 
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import model_from_json

from datagen import get_files_and_labels, scalespec, preprocess, DataGenerator
from learningrate import warmup_cosine_decay, WarmUpCosineDecayScheduler
from specinput import wave_to_mel_spec, load_audio, params

from helpers import remove_file, generate_class_freqs, transfer_data, get_spec_data, evaluate_model


def usage():
    """
    Print usage of the file
    """
    print("Usage: ....")
    return 
    
    

def parse():
    """
    Parse command line arguments 
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:s:a:l:", ["help", "input=", "input_path=", "train_val_split=", "output_spec_path=", "aug=", "loss=", "model_path=", "skip_train", "test_path", "output_test_path"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    input_file_type = None 
    train_val_ratio = None 
    aug_method = None 
    loss_name = None 
    output_spec_path = None 
    model_path = None 
    skip_train = False # By default we do not skip model training 
    test_path = None 
    output_test_path = None 
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file_type = arg
            assert input_file_type in ("audio", "spec")
        elif opt == "--input_path":
            input_data_path = arg
        elif opt in ("-s", "--train_val_split"):
            train_val_ratio = float(arg)
        elif opt in ("-a", "--aug"):
            aug_method = arg 
        elif opt in ("-l", "--loss"):
            loss_name = arg
        elif opt == "--output_spec_path":
            output_spec_path = arg
        elif opt == "--model_path":
            model_path = arg
        elif opt == "--skip_train":
            skip_train = True
        elif opt == "--test_path":
            test_path = arg 
        elif opt == "--output_test_path":
            output_test_path = arg 
        else:
            assert False, "unhandled option"
    
    return input_file_type, input_data_path, train_val_ratio, aug_method, loss_name, output_spec_path, model_path, skip_train, test_path, output_test_path


def train_val_split(data_path, train_split=0.8):
    """    
    Split data into train and vaildation sets by a given percentage
    
    Parameters
    ----------
    data_path: directory of data, should contain two folders p/ and n/ 
    train_split: given percentage of training data size 
    
    Return
    ------
    files_train: list of data files for training set
    files_val: list of data files for validation set
     
    """
    
    class_list = os.listdir(data_path + 'p')

    # generate positive train file paths
    files_train_p, files_val_p, labels = get_files_and_labels(data_path + '/p/',
                                                              train_split=train_split,
                                                              random_state=42,
                                                              classes=class_list)

    # generate negative train file paths
    files_train_n, files_val_n, labels_n = get_files_and_labels(data_path + '/n/',
                                                                train_split=train_split,
                                                                random_state=42,
                                                                classes=class_list) 
    
#     labels_rev = dict((v,k) for (k,v) in labels.items())
    files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
    
    files_train = files_train_p + files_train_n
    files_val = files_val_p + files_val_n
    
    return files_train, files_val, labels


# def generate_aug_data(output_spec_path, files_train, files_val, aug_method):
#     """
#     Generate augmented data with audio augmentation methods
#     """
    
#     # ----- Creat directory -----
#     path_train = path + "/train/"
#     path_val = path + "/val/"
    
#     path_train_p = path + "/train/p/"
#     path_train_n = path + "/train/n/"
#     path_val_p = path + "/val/p/"
#     path_val_n = path + "/val/n/"

#     path_all = [path, path_train, path_val, path_train_p, path_train_n, path_val_p, path_val_n]

#     for p in path_all:
#         if not os.path.exists(p):
#             os.mkdir(p)  
#     # --------------------------
    
#     aug_method_map = {"noise_injection": noise_injection, "shift_time": shift_time, 
#                       "change_pitch": change_pitch, "change_speed": change_speed}
    
#     return 


def generate_spec_data(path, files_train, files_val, aug_method=None):
    """
    Generate spectrogram data with audio data 
    """
    
    # ----- Creat directory -----
    path_train = path + "/train/"
    path_val = path + "/val/"
    
    path_train_p = path + "/train/p/"
    path_train_n = path + "/train/n/"
    path_val_p = path + "/val/p/"
    path_val_n = path + "/val/n/"

    path_all = [path, path_train, path_val, path_train_p, path_train_n, path_val_p, path_val_n]

    for p in path_all:
        if not os.path.exists(p):
            os.mkdir(p)  
    # --------------------------

    print("Generating spectrogram training data ... ")
    files_train_spec = transfer_data(files_train, path_train, mode="wav_to_npy", aug_method=aug_method)
    
    print("Generating spectrogram validation data ... ")
    files_val_spec = transfer_data(files_val, path_val, mode="wav_to_npy", aug_method=aug_method)
    
    return files_train_spec, files_val_spec



def train(files_train, files_val, model_path, loss_name, labels, resize_dim=[224, 224], batch_size=32, 
          params={}):
    """
    Define and train a MobileNetV2 model 
    """

    # train data generator
    train_generator = DataGenerator(files_train,
                                    labels,
                                    resize_dim=resize_dim,
                                    batch_size=batch_size)

    # validation data generator
    val_generator = DataGenerator(files_val,
                                  labels,
                                  resize_dim=resize_dim,
                                  batch_size=batch_size)
    
    num_classes = len(labels)
    
    conv = MobileNetV2(weights='imagenet', 
                   include_top=False, 
                   input_shape=[224, 224, 3])

    for layer in conv.layers:
        layer.trainable = True

    model = models.Sequential()
    model.add(conv)
    model.add(layers.AveragePooling2D((7, 7)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam()

    # note: this loss can be used to avoid assumptions about unlabeled classes
    def masked_loss(y_true, y_pred):
        return K.mean(K.mean(K.binary_crossentropy(tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true),
                                               tf.multiply(y_pred, tf.cast(tf.logical_not(tf.math.is_nan(y_true)), tf.float32))), axis=-1))
    
    # Choose loss function 
    if loss_name == "binary_crossentropy":
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
    elif loss_name == "masked_loss":
        model.compile(loss=masked_loss, optimizer=optimizer)
    else:
        assert False, "unhandled option"
    
    # Print model summary 
    print("---------- Model Summary ----------")
    print(model.summary())
    print("-----------------------------------")
    
    # save model architecture
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        print("Created folder: ", model_path)
    model_out = model_path + "model"
    model_json = model.to_json()
    with open(model_out + '.json', "w") as json_file:
        json_file.write(model_json)
    labels_rev = dict((v,k) for (k,v) in labels.items())
    with open(model_out + '_classes.json', 'w') as f:
        json.dump(labels_rev, f)
    print('Saved model architecture to ', model_path)
    
    # Parameters  
    epochs = params["epochs"]
    warmup_lr = params["warmup_lr"]
    warmup_epochs = params["warmup_epochs"]
    patience = params["patience"]
    steps_per_epoch = len(train_generator)
    base_lr = params["base_lr"]
    hold_base_rate_steps = int(epochs * 0.125 * steps_per_epoch)
    
    total_steps = int(epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    
    # save the best model weights based on validation loss
    val_chkpt = ModelCheckpoint(filepath=model_out+'_best_val.h5',
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1)

    # also save the model weights every 20 epochs
    reg_chkpt = ModelCheckpoint(filepath=model_out+'{epoch:04d}.h5',
                                save_weights_only=True,
                                save_freq=int(steps_per_epoch*20))

    # apply a learning rate schedule
    cosine_warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base= base_lr,
                                                   total_steps= total_steps,
                                                   warmup_learning_rate= warmup_lr,
                                                   warmup_steps= warmup_steps,
                                                   hold_base_rate_steps=hold_base_rate_steps)

    callbacks_list = [val_chkpt, reg_chkpt, cosine_warm_up_lr]

    # Model fitting 
    model_history = model.fit(train_generator,
                          steps_per_epoch = len(train_generator),
                          validation_data = val_generator,
                          epochs = epochs,
                          verbose = 1,
                          callbacks=callbacks_list)

    np.save(model_out + '_history.npy', model_history.history)

    return 



def evaluate(model_path, files_train_spec, files_val_spec, labels, test_path, output_test_path, batch_size=32): 
    
    # Load model 
    model_architecture_path = model_path + '/model.json'
    model_weight_path = model_path + '/model_best_val.h5' # path of model
    model = model_from_json(open(model_architecture_path).read()) # load architecture
    model.load_weights(model_weight_path) # load weights

#     # Load classes 
#     class_dict = json.load(open(model_path+"model_classes.json"))
#     class_list = list(class_dict.values())
#     len(class_list) # Number of classes
    
    # Create data generators 
    resize_dim = [224, 224] # desired shape of generated images
    # train data generator
    train_generator = DataGenerator(files_train_spec,
                                    labels,
                                    resize_dim=resize_dim,
                                    batch_size=batch_size)
    # validation data generator
    val_generator = DataGenerator(files_val_spec,
                                  labels,
                                  resize_dim=resize_dim,
                                  batch_size=batch_size)
    
    # Evaluate on train set 
    accuracy_train, TPR_train, TNR_train, MAP_train, precision_train = evaluate_model(model, train_generator)
    print("overall accuracy on training set:", accuracy_train)
    print("True positive rate (TPR) / Recall on training set: ", TPR_train )
    print("True negative rate (TNR) / Specificity on training set: ", TNR_train )
    print("MAP on training set: ", MAP_train )
    print("Precision on training set: ", precision_train )
    
    # Evaluate on validation set
    accuracy_val, TPR_val, TNR_val, MAP_val, precision_val = evaluate_model(model, val_generator)
    print("overall accuracy on validation set :", accuracy_val)
    print("True positive rate (TPR) / Recall on validation set: ", TPR_val )
    print("True negative rate (TNR) / Specificity on validation set: ", TNR_val )
    print("MAP on validation set: ", MAP_val )
    print("Precision on validation set: ", precision_val )
    
    # Preprocess test data 
    files_test_spec = preprocess_test_data(test_path, output_test_path)
    
    print(files_test_spec)
    
    # Evaluate on test set 
#     batch_size = 2000 # len(files_test)
    # test data generator
    test_generator = TestDataGenerator(files_test_spec,
                                       resize_dim=resize_dim,
                                       batch_size=batch_size,
                                       shuffle=False)

    for batch, _ in tqdm(enumerate(test_generator)):
        pred_prob = model.predict(test_generator[batch][0])
        pred_label = 1 * (pred_prob > 0.5)
        files = [path.split('/')[-1][0:-4] for path in test_generator[batch][1]]
        if batch > 0:
            pred_label_all = np.concatenate((pred_label_all, np.column_stack((files, pred_label))), axis=0)
            pred_prob_all = np.concatenate((pred_prob_all, np.column_stack((files, pred_prob))), axis=0) 
        else:
            pred_label_all = np.column_stack((files, pred_label))
            pred_prob_all = np.column_stack((files, pred_prob))
    
    pred_label_df = pd.DataFrame(pred_label_all, columns =['id'] + class_list)
    pred_label_df.to_csv(model_path+"/prediction_label_test.csv", index=False)
    pred_prob_df = pd.DataFrame(pred_prob_all, columns =['id'] + class_list)
    pred_prob_df.to_csv(model_path+"/prediction_prob_test.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    # Parse command line arguments 
    print("\n")
    print("------------- Starting parsing arguments -------------")
    input_file_type, input_data_path, train_val_ratio, aug_method, loss_name, output_spec_path, model_path, skip_train, test_path, output_test_path = parse() 
    print("------------- Completing parsing arguments -------------")
    print("\n")
    
    
    # Remove .DS_Store files in data directory 
    print("\n")
    print("------------- Starting file removal (Potential unnecessary .DS_Store files generated by Macbook) -------------")
    file_to_remove='.DS_Store'
    remove_file(input_data_path, file_to_remove)
    print("------------- Completing file removal -------------")
    print("\n")    
    
    
    if input_file_type == "audio":
        
        # Generate class-freqs.npy 
        print("\n")
        print("------------- Starting generating class-freqs.npy -------------")
        csv_path = input_data_path + "/class-meta.csv"
        generate_class_freqs(csv_path)
        print("------------- Completing generating class-freqs.npy -------------")
        print("\n")
        
        # Train validation split 
        print("\n")
        print("---------- Starting train validation split ----------")
        files_train, files_val, labels = train_val_split(input_data_path + "/audio/", train_val_ratio)
        print("---------- Completing Train validation split ----------")
        print("\n")

        # Generate image data 
        if aug_method is None:
            # Generate spectrogram data 
            print("\n")
            print("---------- Start generating spectrogram data ----------")
            files_train_spec, files_val_spec = generate_spec_data(output_spec_path, files_train, files_val)
            print("---------- Complete generating spectrogram data ----------")
            print("\n")
        else:
            # Generate augmented data 
            print("\n")
            print("---------- Start generating augmented data ----------")
            print("Augmentation methods: ", aug_method)
            files_train_spec, files_val_spec = generate_spec_data(output_spec_path, files_train, files_val, aug_method)
            print("---------- Complete generating augmented data ----------")
            print("\n")
    
    elif input_file_type == "spec":
        print("\n")
        print("---------- Start getting spectrogram data ----------")
        files_train_spec, files_val_spec, labels = get_spec_data(input_data_path)
        print("---------- Complete getting spectrogram data ----------")
        print("\n")
    
    else:
        assert False, "unhandled input data type"
        

    if not skip_train: 
        # Train model 
        params = {"epochs": 100, 
                  "warmup_lr": 1e-5, 
                  "warmup_epochs": 10, 
                  "patience": 10, 
                  "base_lr": 0.0015}
        print("\n")
        print("---------- Start training model ----------")
        train(files_train_spec, files_val_spec, model_path, loss_name, labels, params=params)
        print("---------- Complete training model ----------")
        print("\n")

    
    # Evaluate model 
    print("\n")
    print("---------- Start evaluating model ----------")
    evaluate(model_path, files_train_spec, files_val_spec, labels, test_path, output_test_path)
    print("---------- Complete evaluating model ----------")


