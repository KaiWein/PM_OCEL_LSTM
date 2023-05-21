import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import LSTM, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def LSTM_MODEL(X, y_a, y_t, y_tr,filename):
    print('Build model...')
    number_of_train_cases, max_trace_length, num_of_features = X.shape
    target_act_length = y_a.shape[1]
    main_input = Input(shape=(max_trace_length, num_of_features), name='main_input')
    # train a 2-layer LSTM with one shared layer
    # the shared layer
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)  
    b1 = tf.keras.layers.BatchNormalization()(l1)
     # the layer specialized in activity prediction
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) 
    b2_1 = tf.keras.layers.BatchNormalization()(l2_1)
    # the layer specialized in time prediction
    l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)  
    b2_2 = tf.keras.layers.BatchNormalization()(l2_2)
    # the layer specialized in time remaining prediction
    l2_3 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)  
    b2_3 = tf.keras.layers.BatchNormalization()(l2_3)

    act_output = Dense(target_act_length, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

    timeR_output = Dense(1, kernel_initializer='glorot_uniform', name='timeR_output')(b2_3)

    model = Model(inputs=[main_input], outputs=[act_output, time_output, timeR_output])

    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae', 'timeR_output': 'mae'}, optimizer='nadam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('./output_files/models/model_'+filename+'.h5', monitor='val_loss',
                                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=0, mode='auto', min_delta=0.0001,
                                cooldown=0, min_lr=0)

    history = model.fit(X, {'act_output': y_a, 'time_output': y_t, 'timeR_output': y_tr}, validation_split=0.2, verbose=2,
            callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_trace_length, epochs=500)
    # list all data in history
    return history, model_checkpoint
