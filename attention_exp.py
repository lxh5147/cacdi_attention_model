'''
Created on Jul 13, 2016

@author: lxh5147
'''
from attention_model import  build_classifier_with_hierarchical_attention,apply_mlp_softmax_classifier
from attention_layer import check_and_throw_if_fail, HierarchicalAttention
from keras import backend as K
import numpy as np
from keras.layers import Input
from keras.callbacks import EarlyStopping

import tensorflow as tf

K.set_session(tf.Session())
sess = K.get_session()

              
def fake_data(input_shape, dtype='float32', max_int=10): 
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

def faked_dataset(inputs, total, timesteps, vocabulary_size,output_dim):
    input_vals = []
    for input_tensor in inputs:
        input_val=fake_data((total,) + K.int_shape(input_tensor)[1:],K.dtype(input_tensor), max_int=vocabulary_size)
        input_vals.append(input_val)    
    output_val = fake_data((total, timesteps) , 'int32', max_int=output_dim)
    output_val = output_val.reshape((total*timesteps,))
    output_val = to_binary_matrix(output_val, max_int=output_dim)
    output_val = output_val.reshape((total,timesteps,output_dim))
    return input_vals, output_val

def debug_attention_layer():
    input_shape=(7,8,5,6,9)
    #record, document,section,sentence,word
    input_feature_dims=(20,10,50,60,30)
    #document, section, sentence, word
    attention_output_dims=(45,35,25,65)
    #document, section, sentence, word
    attention_weight_vector_dims = (82,72,62,52)
    #embedding
    embedding_rows = 200
    embedding_dim = 50
    #classifier    
    initial_embedding = np.random.random((embedding_rows,embedding_dim))
    inputs= HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
    hierarchical_attention = HierarchicalAttention(attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder = False)
    output = hierarchical_attention(inputs)

    total = 2
    output_dim = 10
    timesteps=input_shape[0]          
    x_train, _ =  faked_dataset(inputs, total, timesteps, embedding_rows,output_dim)
    
    #build feed_dic
    feed_dict = {}
    for i in range(len(inputs)):
        feed_dict[inputs[i]] = x_train[i]
    feed_dict[K.learning_phase()] = 1
    #tf.initialize_all_variables()
    #y_out is fine 2,7, 110
    y_out = sess.run(output, feed_dict=feed_dict) 
    check_and_throw_if_fail(y_out.shape==(2,7,110),"y_out")
    
def debug_softmax_layer():
    input_sequence = Input(shape=(7,110))
    output_dim = 5
    hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
    drop_out_rates = ( 0.5, 0.6)     
  
    output = apply_mlp_softmax_classifier(input_sequence, output_dim, hidden_unit_numbers, drop_out_rates)
    #build feed_dic
    total = 4
    feed_dict = {}
    feed_dict[input_sequence] = np.random.random((total,) + K.int_shape(input_sequence)[1:] )
    feed_dict[K.learning_phase()] = 1
    #tf.initialize_all_variables()
    #y_out is fine 2,7, 110
    y_out = sess.run(output, feed_dict=feed_dict) 
    check_and_throw_if_fail(y_out.shape==(total, K.int_shape(input_sequence)[1], output_dim ) ,"y_out")

def debug_attention_with_classifier_layer():
    input_shape=(7,8,5,6,9)
    #record, document,section,sentence,word
    input_feature_dims=(20,10,50,60,30)
    #document, section, sentence, word
    attention_output_dims=(45,35,25,65)
    #document, section, sentence, word
    attention_weight_vector_dims = (82,72,62,52)
    #embedding
    embedding_rows = 200
    embedding_dim = 50
    output_dim = 5
    #hidden_unit_numbers=(5, 20) # 5--> first hidden layer, 20 --> second hidden layer
    #drop_out_rates = ( 0.5, 0.6) 
    hidden_unit_numbers=() # 5--> first hidden layer, 20 --> second hidden layer
    drop_out_rates = () 
    
    #classifier    
    initial_embedding = np.random.random((embedding_rows,embedding_dim))
    inputs= HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
    hierarchical_attention = HierarchicalAttention(attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder = False)
    output = hierarchical_attention(inputs)
    output = apply_mlp_softmax_classifier(output, output_dim, hidden_unit_numbers, drop_out_rates)
    total = 2
    timesteps=input_shape[0]          
    x_train, _ =  faked_dataset(inputs, total, timesteps, embedding_rows,output_dim)
    
    #build feed_dic
    feed_dict = {}
    for i in range(len(inputs)):
        feed_dict[inputs[i]] = x_train[i]
    feed_dict[K.learning_phase()] = 1

    y_out = sess.run(output, feed_dict=feed_dict) 
    check_and_throw_if_fail(y_out.shape==(total,timesteps,output_dim),"y_out")
    
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
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #train 
    total = 4
    batch_size= 2
    nb_epoch = 5
    timesteps=input_shape[0]
    
    x_train, y_train =  faked_dataset(model.inputs, total, timesteps, vocabulary_size,output_dim)
    model.fit(x_train, y_train,  batch_size, nb_epoch, verbose=1, callbacks=[EarlyStopping(patience=5)],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None)
    #evaluate
    total = 4
    x_test, y_test =  faked_dataset(model.inputs, total, timesteps, vocabulary_size,output_dim)
    model.evaluate(x_test, y_test, batch_size=2, verbose=1, sample_weight=None)
    #predict
    total = 2
    x_pred, _ =  faked_dataset(model.inputs, total, timesteps, vocabulary_size,output_dim)
    y_pred= model.predict(x_pred, batch_size=1,  verbose=1)
    check_and_throw_if_fail(y_pred.shape == (total, timesteps, output_dim),"y_pred")
       
if __name__ == '__main__':
    faked_exp()
    #debug_softmax_layer()
    #debug_attention_with_classifier_layer()
    #debug_attention_layer()