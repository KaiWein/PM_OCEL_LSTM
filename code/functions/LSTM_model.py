import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import LSTM, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def LSTM_MODEL(X, y_a, y_t, y_tr, filename):
    print('Build model...')
    number_of_train_cases, max_trace_length, num_of_features = X.shape
    target_act_length = y_a.shape[1]
    
    main_input = Input(shape=(max_trace_length, num_of_features), name='main_input')
    
    # Shared LSTM layer
    shared_lstm = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)
    shared_lstm = tf.keras.layers.BatchNormalization()(shared_lstm)
    
    # Activity prediction LSTM
    act_lstm = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm)
    act_lstm = tf.keras.layers.BatchNormalization()(act_lstm)
    act_output = Dense(target_act_length, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(act_lstm)
    
    # Time prediction LSTM
    time_lstm = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm)
    time_lstm = tf.keras.layers.BatchNormalization()(time_lstm)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(time_lstm)
    
    # Time remaining prediction LSTM
    timeR_lstm = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm)
    timeR_lstm = tf.keras.layers.BatchNormalization()(timeR_lstm)
    timeR_output = Dense(1, kernel_initializer='glorot_uniform', name='timeR_output')(timeR_lstm)
    
    model = Model(inputs=[main_input], outputs=[act_output, time_output, timeR_output])

    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae', 'timeR_output': 'mae'}, optimizer='nadam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    model_checkpoint = ModelCheckpoint('./output_files/models/model_'+filename+'_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=0, mode='auto', min_delta=0.0001,
                                   cooldown=0, min_lr=0)

    history = model.fit(X, {'act_output': y_a, 'time_output': y_t, 'timeR_output': y_tr}, validation_split=0.2, verbose=2,
                        callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_trace_length, epochs=500)
    
    # List all data in history
    return history, model_checkpoint, early_stopping
