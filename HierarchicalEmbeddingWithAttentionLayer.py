'''
Created on Jul 5, 2016

@author: lxh5147
'''
from keras import backend as K


from engine import ModelEx as Model

from keras.layers import (Lambda,
                          Dropout,
                          Convolution1D,
                          Input,
                          BatchNormalization,
                          merge)

from attention_layer import (ComposedLayer,ManyToOnePooling,
                             mask_sequence,
                             MLPClassifierLayer,
                             remove_mask,
                             add_mask,
                             SequenceToSequenceEncoder,
                             Attention,
                             check_and_throw_if_fail,
                             get_copy_of,
                             Embedding)

class HierarchicalEmbeddingWithAttention(ComposedLayer):
    '''
    nb_layers = number of non-leaf layers

    Indices_level0     Indices_levelN
    Embedding_level0   Embedding_levelN   Word_Embedding
    Indices of level i includes children of level i+1
    Embedding level i  embedding of level i units
    Embedding level i tensor shape:  nb_samples, embedding_dim
    Indices_level i   tensor shape: nb_samples, max_nb_children
    Each non-leaf level has an attention mechanism to aggregate its children to a dense tensor
    '''

    def __init__(self,
                 sparse_feature_dims,
                 attention_output_dims,
                 attention_weight_vector_dims,
                 embedding_rows,
                 embedding_dim,
                 initial_embedding=None,
                 use_cnn_as_sequence_to_sequence_encoder=False,
                 input_window_sizes=None,
                 pooling_mode=None,
                 norm_inner_output = False,
                 hierarchical_layer_dropout_W=0.,
                 hierarchical_layer_dropout_U=0.,
                 attention_input_dropout=0.,
                 attention_dropout=0.,
                 weight_regularizer_batch_norm = None,
                 weight_regularizer_proj = None,
                 weight_regularizer_encoder = None,
                 weight_regularizer_attention = None,
                 update_embedding = True,
                 proj_learning_rate = None,
                 **kwargs):

        check_and_throw_if_fail(len(sparse_feature_dims) > 0, "sparse_feature_dims")
        check_and_throw_if_fail(len(attention_output_dims) >= 0, "attention_output_dims")
        check_and_throw_if_fail(len(sparse_feature_dims) == len(attention_output_dims) + 1, "tree_shape")
        check_and_throw_if_fail(pooling_mode or len(attention_weight_vector_dims) == len(attention_output_dims),
                                "attention_weight_vector_dims")
        check_and_throw_if_fail(use_cnn_as_sequence_to_sequence_encoder == False or input_window_sizes is not None,
                                "input_window_sizes")
        check_and_throw_if_fail(input_window_sizes is None or len(input_window_sizes) == len(attention_output_dims),
                                "input_window_sizes")
        super(HierarchicalEmbeddingWithAttention, self).__init__(**kwargs)
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims = attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding = initial_embedding
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.sparse_feature_dims = sparse_feature_dims
        self.pooling_mode = pooling_mode
        self.supports_masking = True
        self.norm_inner_output = norm_inner_output
        self.hierarchical_layer_dropout_W = hierarchical_layer_dropout_W
        self.hierarchical_layer_dropout_U = hierarchical_layer_dropout_U
        self.attention_input_dropout = attention_input_dropout
        self.attention_dropout = attention_dropout
        self.weight_regularizer_batch_norm = weight_regularizer_batch_norm
        self.weight_regularizer_proj = weight_regularizer_proj
        self.weight_regularizer_encoder = weight_regularizer_encoder
        self.weight_regularizer_attention = weight_regularizer_attention
        self.update_embedding = update_embedding
        self.proj_learning_rate = proj_learning_rate
        self._build_layers()

    def _build_layers(self):
        # word embedding
        self.embedding = Embedding(self.embedding_rows,
                                   self.embedding_dim,
                                   weights=[self.initial_embedding],
                                   W_regularizer=get_copy_of(self.weight_regularizer_proj),
                                   update_weights= self.update_embedding,
                                   learning_rate = self.proj_learning_rate)

        self.embedding.build(input_shape=None)

        self.norm_layers = []
        self.attention_layers = []
        self.encoder_layers = []
        self.output_dims = []
        total_level = len(self.sparse_feature_dims)
        #0 -> word ->....-> snapshot
        for cur_level in xrange(total_level - 1):
            cur_output_dim = self.attention_output_dims[cur_level]

            if self.pooling_mode:
                attention_weight_vector_dim = None
            else:
                attention_weight_vector_dim = self.attention_weight_vector_dims[cur_level]
            if self.use_cnn_as_sequence_to_sequence_encoder:
                cur_window_size = self.input_window_sizes[cur_level]
            else:
                cur_window_size = None
            norm_layer, attention_layer, encoder_layer,total_output_dim = \
                self.create_attention_layer(attention_weight_vector_dim,
                                            cur_output_dim,
                                            cur_window_size)
            self.norm_layers.append(norm_layer)
            self.attention_layers.append(attention_layer)
            self.encoder_layers.append(encoder_layer)
            self.output_dims.append(total_output_dim)

            if self.use_cnn_as_sequence_to_sequence_encoder:
                dropout_layer = Dropout(p=self.hierarchical_layer_dropout_W)
                # add to layers of current container
                self._layers.append(dropout_layer)
                # associate dropout to the encoder
                encoder_layer.dropout = dropout_layer

        self._layers += ([self.embedding] + self.norm_layers + self.attention_layers + self.encoder_layers)

    def create_attention_layer(
            self,
            attention_weight_vector_dim,
            cur_output_dim,
            cur_window_size):

        norm = BatchNormalization(mode=1,
                                  gamma_regularizer=get_copy_of(self.weight_regularizer_batch_norm),
                                  beta_regularizer=get_copy_of(self.weight_regularizer_batch_norm))
        if self.pooling_mode:  # max, sum or None
            attention = ManyToOnePooling(mode=self.pooling_mode)
        else:
            attention = Attention(
                attention_weight_vector_dim,
                attention_dropout=self.attention_dropout,
                attention_input_dropout=self.attention_input_dropout,
                weight_regularizer=get_copy_of(self.weight_regularizer_attention))

        if self.use_cnn_as_sequence_to_sequence_encoder:
            total_output_dim = cur_output_dim
            cnn = Convolution1D(cur_output_dim,
                                filter_length=cur_window_size,
                                border_mode='same',
                                activation="relu",
                                W_regularizer=get_copy_of(self.weight_regularizer_encoder),
                                b_regularizer=get_copy_of(self.weight_regularizer_encoder))
            return norm, attention, cnn, total_output_dim
        else:
            total_output_dim = cur_output_dim
            seq2seq = SequenceToSequenceEncoder(
                cur_output_dim,
                dropout_W=self.hierarchical_layer_dropout_W,
                dropout_U=self.hierarchical_layer_dropout_U,
                W_regularizer=get_copy_of(self.weight_regularizer_encoder),
                U_regularizer=get_copy_of(self.weight_regularizer_encoder),
                b_regularizer=get_copy_of(self.weight_regularizer_encoder))

            if seq2seq.is_bi_directional:
                total_output_dim = total_output_dim*2
            return norm, attention, seq2seq, total_output_dim

    def call_attention_layer(self,
                             input_sparse_features,
                             input_sequence,
                             input_embedding,
                             input_embedding_dim,
                             norm_layer,
                             attention_layer,
                             encoder_layer,
                             input_mask = None):
        #remove mask and then apply to embedding
        # nb_samples, time_steps
        input_sequence = remove_mask(input_sequence)
        input_sequence_embedding = Lambda (
            lambda  x: K.gather(input_embedding, input_sequence),
            output_shape=lambda  input_shape: input_shape + (input_embedding_dim,))\
            (input_sequence)

        if input_sparse_features is not None:
            input_sequence_embedding = merge(inputs=[input_sparse_features,
                                                      input_sequence_embedding],
                                              mode='concat')

        if self.norm_inner_output:
            input_sequence_embedding = norm_layer(input_sequence_embedding)
        # mask the input sequence
        input_sequence_embedding_with_mask = add_mask(input_sequence_embedding, input_mask)
        if self.use_cnn_as_sequence_to_sequence_encoder:
            assert encoder_layer.dropout
            input_sequence_embedding_with_mask = encoder_layer.dropout(input_sequence_embedding_with_mask)
            # since conv does not support mask, remove mask first
            input_sequence_embedding = remove_mask(input_sequence_embedding_with_mask)
            output = encoder_layer(input_sequence_embedding)
            output_with_mask = add_mask(output,input_mask)
            attention_vector = attention_layer(output_with_mask)
        else:
            attention_vector =  attention_layer(encoder_layer(input_sequence_embedding_with_mask))

        return attention_vector

    def call(self, inputs, mask=None):
        '''
        inputs: a list of inputs; the first layer is lowest level sequence, second layer lowest level input features, ..., the top level input features
        returns a tensor of shape: batch_size*snapshots*output_dim
        '''
        if type(inputs) is not list:
            inputs = [inputs]

        if mask is not None and type(mask) is not list:
            mask = [mask]

        total_level = len(self.sparse_feature_dims)
        to_sparse_feature_input ={}
        total_sparse_features = 0
        for i,sparse_feature_dim in enumerate(self.sparse_feature_dims):
            if sparse_feature_dim > 0:
                to_sparse_feature_input[i] = total_sparse_features
                total_sparse_features += 1
            else:
                to_sparse_feature_input[i]=-1

        total_input = total_level - 1 + total_sparse_features
        check_and_throw_if_fail(len(inputs) == total_input, 'inputs')

        embedding = self.embedding.W
        embedding_dim=self.embedding_dim

        i = 0

        for norm_layer, attention_layer, encoder_layer, total_output_dim  in zip(self.norm_layers,self.attention_layers, self.encoder_layers,self.output_dims):
            input_sequence = inputs[i]
            if mask is not None:
                input_mask = mask[i]
            else:
                input_mask = None

            if to_sparse_feature_input[i] >=0:
                sparse_feature = inputs[total_level-1 + to_sparse_feature_input[i]]
            else:
                sparse_feature = None

            embedding = self.call_attention_layer(
                             input_sparse_features =sparse_feature ,
                             input_sequence = input_sequence,
                             input_embedding = embedding,
                             input_embedding_dim = embedding_dim,
                             norm_layer = norm_layer,
                             attention_layer = attention_layer,
                             encoder_layer = encoder_layer,
                             input_mask = input_mask)
            embedding_dim = total_output_dim

            i+=1

        if self.sparse_feature_dims[-1] > 0:
            sparse_feature = inputs[-1]
            embedding = merge(inputs=[sparse_feature, embedding], mode='concat')

        return embedding

    @staticmethod
    def build_inputs(input_feature_dims):
        '''
        Every layer, except the top layer, has a sequence input;
        Every layer has a sparse feature tensor as input
        '''
        inputs = []
        total_level = len(input_feature_dims)
        # sequence input tensor
        for _ in xrange(total_level-1):
            input = Input(shape=(None,), dtype="int32")
            inputs.append(input)
        # sparse feature input tensor
        for input_feature_dim in input_feature_dims[:-1]:
            if input_feature_dim > 0:
                input = Input(shape=(None,input_feature_dim))
                inputs.append(input)

        # the last sparse feature input tensor
        if input_feature_dims[-1] > 0:
            input = Input(shape=(input_feature_dims[-1],))
            inputs.append(input)

        return inputs

    def compute_mask(self, inputs, mask):
        # no mask
        return None

    def get_output_shape_for(self, input_shapes):
        output_dim = self.sparse_feature_dims[-1]
        # plus the internal attention output dim if applicable
        if self.output_dims:
            output_dim += self.output_dims[-1]
        return (None, output_dim)

    def get_config(self):
        config = {'sparse_feature_dims': self.sparse_feature_dims,
                  'attention_output_dims': self.attention_output_dims,
                  'attention_weight_vector_dims': self.attention_weight_vector_dims,
                  'embedding_rows': self.embedding_rows,
                  'embedding_dim': self.embedding_dim,
                  'initial_embedding': self.initial_embedding,
                  'use_cnn_as_sequence_to_sequence_encoder': self.use_cnn_as_sequence_to_sequence_encoder,
                  'input_window_sizes': self.input_window_sizes,
                  'pooling_mode': self.pooling_mode,
                  'norm_inner_output': self.norm_inner_output,
                  'hierarchical_layer_dropout_W':self.hierarchical_layer_dropout_W,
                  'hierarchical_layer_dropout_U': self.hierarchical_layer_dropout_U,
                  'attention_input_dropout': self.attention_input_dropout,
                  'attention_dropout': self.attention_dropout,
                  'weight_regularizer_batch_norm': self.weight_regularizer_batch_norm,
                  'weight_regularizer_proj':self.weight_regularizer_proj,
                  'weight_regularizer_encoder':self.weight_regularizer_encoder,
                  'weight_regularizer_attention':self.weight_regularizer_attention,
                  'update_embedding':self.update_embedding,
                  'proj_learning_rate':self.proj_learning_rate,
                  }
        base_config = super(HierarchicalEmbeddingWithAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClassifierWithHierarchicalEmbeddingAttention(ComposedLayer):
    def __init__(self,
                 input_feature_dims,
                 attention_output_dims,
                 attention_weight_vector_dims,
                 embedding_rows,
                 embedding_dim,
                 initial_embedding,
                 output_dim,
                 hidden_unit_numbers,
                 hidden_unit_activation_functions,
                 output_activation_function='softmax',
                 use_cnn_as_sequence_to_sequence_encoder=False,
                 input_window_sizes=None,
                 pooling_mode=None,
                 mlp_softmax_classifier_input_drop_out_rate = 0.,
                 hierarchical_layer_dropout_W=0.,
                 hierarchical_layer_dropout_U=0.,
                 attention_input_dropout=0.,
                 attention_dropout=0.,
                 mlp_classifier_norm_inner_output = False,
                 attention_norm_inner_output=False,
                 weight_regularizer_batch_norm = None,
                 weight_regularizer_hidden = None,
                 weight_regularizer_proj = None,
                 weight_regularizer_encoder = None,
                 weight_regularizer_attention = None,
                 weight_regularizer_mlp_output = None,
                 update_embedding = True,
                 proj_learning_rate = None,
                 **kwargs):
        super(ClassifierWithHierarchicalEmbeddingAttention, self).__init__(**kwargs)

        self.input_feature_dims = input_feature_dims
        self.attention_output_dims = attention_output_dims
        self.attention_weight_vector_dims = attention_weight_vector_dims
        self.embedding_rows = embedding_rows
        self.embedding_dim = embedding_dim
        self.initial_embedding = initial_embedding
        self.output_dim = output_dim
        self.hidden_unit_numbers = hidden_unit_numbers
        self.hidden_unit_activation_functions = hidden_unit_activation_functions
        self.output_activation_function = output_activation_function
        self.use_cnn_as_sequence_to_sequence_encoder = use_cnn_as_sequence_to_sequence_encoder
        self.input_window_sizes = input_window_sizes
        self.pooling_mode = pooling_mode
        self.supports_masking = True
        self.mlp_softmax_classifier_input_drop_out_rate = mlp_softmax_classifier_input_drop_out_rate
        self.hierarchical_layer_dropout_W=hierarchical_layer_dropout_W
        self.hierarchical_layer_dropout_U=hierarchical_layer_dropout_U
        self.attention_input_dropout = attention_input_dropout
        self.attention_dropout = attention_dropout
        self.mlp_classifier_norm_inner_output = mlp_classifier_norm_inner_output
        self.attention_norm_inner_output = attention_norm_inner_output
        self.weight_regularizer_batch_norm = weight_regularizer_batch_norm
        self.weight_regularizer_hidden = weight_regularizer_hidden
        self.weight_regularizer_proj = weight_regularizer_proj
        self.weight_regularizer_encoder = weight_regularizer_encoder
        self.weight_regularizer_attention = weight_regularizer_attention
        self.weight_regularizer_mlp_output = weight_regularizer_mlp_output
        self.update_embedding = update_embedding
        self.proj_learning_rate = proj_learning_rate
        self._build_layers()

    def _build_layers(self):
        # only consider this weight regularizer as a template
        self.hierarchical_attention = HierarchicalEmbeddingWithAttention(
                                                            self.input_feature_dims,
                                                            self.attention_output_dims,
                                                            self.attention_weight_vector_dims,
                                                            self.embedding_rows, self.embedding_dim,
                                                            self.initial_embedding,
                                                            self.use_cnn_as_sequence_to_sequence_encoder,
                                                            self.input_window_sizes, self.pooling_mode,
                                                            norm_inner_output=self.attention_norm_inner_output,
                                                            hierarchical_layer_dropout_W=self.hierarchical_layer_dropout_W,
                                                            hierarchical_layer_dropout_U=self.hierarchical_layer_dropout_U,
                                                            attention_input_dropout=self.attention_input_dropout,
                                                            attention_dropout=self.attention_dropout,
                                                            weight_regularizer_batch_norm=get_copy_of(self.weight_regularizer_batch_norm),
                                                            weight_regularizer_proj = get_copy_of(self.weight_regularizer_proj),
                                                            weight_regularizer_encoder = get_copy_of(self.weight_regularizer_encoder),
                                                            weight_regularizer_attention=get_copy_of(self.weight_regularizer_attention),
                                                            update_embedding=self.update_embedding,
                                                            proj_learning_rate=self.proj_learning_rate,)


        self.mlp_softmax_classifier = MLPClassifierLayer(self.output_dim,
                                                         self.hidden_unit_numbers,
                                                         self.hidden_unit_activation_functions,
                                                         self.output_activation_function,
                                                         norm_inner_output=self.mlp_classifier_norm_inner_output,
                                                         use_sequence_input = False,
                                                         weight_regularizer_batch_norm = get_copy_of(self.weight_regularizer_batch_norm),
                                                         weight_regularizer_hidden=get_copy_of(self.weight_regularizer_hidden),
                                                         weight_regularizer_mlp_output = get_copy_of(self.weight_regularizer_mlp_output),
                                                         input_drop_out_rate = self.mlp_softmax_classifier_input_drop_out_rate)

        self._layers += [self.hierarchical_attention,
                         self.mlp_softmax_classifier]

    def call(self, inputs, mask=None):
        output = self.hierarchical_attention(inputs)
        # shape of output: nb_samples, time_steps, output_dim
        return self.mlp_softmax_classifier(output)

    def compute_mask(self, inputs, mask):
        return None

    def get_output_shape_for(self, input_shapes):
        return (None, self.output_dim)

    def get_config(self):
        config = {
                  'input_feature_dims': self.input_feature_dims,
                  'attention_output_dims': self.attention_output_dims,
                  'attention_weight_vector_dims': self.attention_weight_vector_dims,
                  'embedding_rows': self.embedding_rows,
                  'embedding_dim': self.embedding_dim,
                  'initial_embedding': self.initial_embedding,
                  'output_dim': self.output_dim,
                  'hidden_unit_numbers': self.hidden_unit_numbers,
                  'hidden_unit_activation_functions': self.hidden_unit_activation_functions,
                  'output_activation_function': self.output_activation_function,
                  'use_cnn_as_sequence_to_sequence_encoder': self.use_cnn_as_sequence_to_sequence_encoder,
                  'input_window_sizes': self.input_window_sizes,
                  'pooling_mode': self.pooling_mode,
                  'mlp_softmax_classifier_input_drop_out_rate':self.mlp_softmax_classifier_input_drop_out_rate,
                  'mlp_classifier_norm_inner_output':self.mlp_classifier_norm_inner_output,
                  'attention_norm_inner_output': self.attention_norm_inner_output,
                  'hierarchical_layer_dropout_W': self.hierarchical_layer_dropout_W,
                  'hierarchical_layer_dropout_U': self.hierarchical_layer_dropout_U,
                  'attention_input_dropout': self.attention_input_dropout,
                  'attention_dropout': self.attention_dropout,
                   'weight_regularizer_batch_norm':self.weight_regularizer_batch_norm,
                  'weight_regularizer_hidden':self.weight_regularizer_hidden,
                  'weight_regularizer_proj':self.weight_regularizer_proj,
                  'weight_regularizer_encoder':self.weight_regularizer_encoder,
                  'weight_regularizer_attention':self.weight_regularizer_attention,
                  'weight_regularizer_mlp_output':self.weight_regularizer_mlp_output,
                  'update_embedding':self.update_embedding,
                  'proj_learning_rate': self.proj_learning_rate,
                }

        base_config = super(ClassifierWithHierarchicalEmbeddingAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def build_classifier_with_hierarchical_embedding_attention(
        input_feature_dims,
        attention_output_dims,
        attention_weight_vector_dims,
        embedding_rows,
        embedding_dim,
        initial_embedding,
        output_dim,
        hidden_unit_numbers,
        hidden_unit_activation_functions,
        output_activation_function="softmax",
        use_cnn_as_sequence_to_sequence_encoder = False,
        input_window_sizes = None,
        pooling_mode = None,
        mlp_softmax_classifier_input_drop_out=0.,
        hierarchical_layer_dropout_W=0.,
        hierarchical_layer_dropout_U=0.,
        attention_dropout=0.,
        attention_input_dropout=0.,
        mlp_classifier_norm_inner_output=False,
        attention_norm_inner_output = False,
        input_mask_value = -1, # use negative or any impossible index as mask value
        on_train_on_batch_failed = None,
        weight_regularizer_batch_norm = None,
        weight_regularizer_hidden=None, # config of a regularizer, used to create instances
        weight_regularizer_proj = None,
        weight_regularizer_encoder = None,
        weight_regularizer_attention = None,
        weight_regularizer_mlp_output = None,
        update_embedding = True,
        proj_learning_rate = None,
    ):

    #input_feature_dims: word, sentence, ..., snapshot
    inputs = HierarchicalEmbeddingWithAttention.build_inputs(input_feature_dims)
    total_sequence_input = len(input_feature_dims)-1

    #  calculate mask, apply mask and attach mask
    inputs_with_masks = [mask_sequence(input, mask_value=input_mask_value) for input in inputs[:total_sequence_input]] + \
             [input  for input in inputs[total_sequence_input:]]

    classifier = ClassifierWithHierarchicalEmbeddingAttention (
        input_feature_dims,
        attention_output_dims,
        attention_weight_vector_dims,
        embedding_rows,
        embedding_dim,
        initial_embedding,
        output_dim, hidden_unit_numbers,
        hidden_unit_activation_functions,
        output_activation_function,
        use_cnn_as_sequence_to_sequence_encoder,
        input_window_sizes,
        pooling_mode,
        mlp_softmax_classifier_input_drop_out,
        hierarchical_layer_dropout_W=hierarchical_layer_dropout_W,
        hierarchical_layer_dropout_U=hierarchical_layer_dropout_U,
        attention_dropout=attention_dropout,
        attention_input_dropout=attention_input_dropout,
        mlp_classifier_norm_inner_output=mlp_classifier_norm_inner_output,
        attention_norm_inner_output = attention_norm_inner_output,
        weight_regularizer_batch_norm = weight_regularizer_batch_norm,
        weight_regularizer_hidden=weight_regularizer_hidden,
        weight_regularizer_proj = weight_regularizer_proj,
        weight_regularizer_encoder = weight_regularizer_encoder,
        weight_regularizer_attention = weight_regularizer_attention,
        weight_regularizer_mlp_output = weight_regularizer_mlp_output,
        update_embedding=update_embedding,
        proj_learning_rate = proj_learning_rate,)

    output = classifier(inputs_with_masks)

    # Note: the Keras model use the first input to calculate batch size.
    # Internally, the first input corresponds to word sequence,
    # and the last corresponds to document sequence.
    # Similarly, the first sparse input corresponds to word features,
    # and the last sparse input corresponds to review features.
    # We reverse the order of sequence inputs and sparse inputs,
    # so that keras can calculate the batch size correctly
    # And data provider should adjust order of input values accordingly.

    sequence_inputs = inputs[:total_sequence_input]
    sequence_inputs.reverse()
    sparse_inputs = inputs[total_sequence_input:]
    sparse_inputs.reverse()
    model = Model(input = sequence_inputs + sparse_inputs,
                  output = output,
                  on_train_on_batch_failed = on_train_on_batch_failed)
    # inject a property of the last output layer
    model.mlp_softmax_classifier = classifier.mlp_softmax_classifier
    return model