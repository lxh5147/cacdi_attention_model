'''
Created on Jul 13, 2016

@author: lxh5147
'''
from attention_model import  build_classifier_with_hierarchical_attention
from keras import backend as K
import numpy as np

def fake_data(input_shape, total,dtype='float32', max_int=10):  
    input_shape[0]=total
    val = np.random.random(input_shape)
    if dtype == 'int32':
        val = val * max_int
        val = val.astype(int)
    return val

def to_binary_matrix(x,max_int):
    '''
    x: a list of positions, whose value is not zero
    y: one hot representation
    '''
    len_x = len(x)
    y = np.zeros((len_x, max_int))
    y[np.arange(len_x),x] = 1
    return y

def faked_dataset(model, total, timesteps, vocabulary_size,output_dim):
    input_vals = []
    for input_tensor in model.inputs:
        input_val=fake_data(K.int_shape(input_tensor), total,K.dtype(input_tensor), max_int=vocabulary_size)
        input_vals.append(input_val)    
    output_val = fake_data((total, timesteps) , total,'int32', max_int=output_dim)
    output_val = output_val.reshape((total*timesteps,))
    output_val = to_binary_matrix(output_val, max_int=output_dim)
    return input_vals, output_val

def faked_exp():
    # time_steps* documents * sections* sentences * words
    input_shape=(7,8,5,6,9)
    #record, document,section,sentence,word
    input_feature_dims=(20,10,50,60,30)
    #document, section, sentence, word
    output_dims=(45,35,25,65)
    #document, section, sentence, word
    attention_weight_vector_dims = (82,72,62,52)
    #embedding
    vocabulary_size = 200
    word_embedding_dim = 50
    #classifier
    output_dim = 5
    hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
    drop_out_rates = ( 0.5, 0.6) 
    use_sequence_to_vector_encoder = False 
    
    initial_embedding = np.random.random((vocabulary_size,word_embedding_dim))
    model = build_classifier_with_hierarchical_attention(input_shape, input_feature_dims, output_dims, attention_weight_vector_dims, vocabulary_size, word_embedding_dim, initial_embedding, use_sequence_to_vector_encoder, output_dim, hidden_unit_numbers, drop_out_rates)
    #compile the model
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])
    #train 
    total = 20
    batch_size= 10
    nb_epoch = 5
   
    x_train, y_train =  faked_dataset(model, total, timesteps=input_shape[0], vocabulary_size,output_dim)
    
    model.fit(x_train, y_train,  batch_size, nb_epoch, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None)
    #evaluate
    total = 4
    x_test, y_test =  faked_dataset(model, total, timesteps=input_shape[0], vocabulary_size,output_dim)
    model.evaluate(x_test, y_test, batch_size=2, verbose=1, sample_weight=None)
    #predict
    total = 2
    x_pred, _ =  faked_dataset(model, total, timesteps=input_shape[0], vocabulary_size,output_dim)
    y_pred= model.predict(x_pred, batch_size=1,  verbose=1)
    print(y_pred)
       
if __name__ == '__main__':
    faked_exp()