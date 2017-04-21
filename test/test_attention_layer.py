'''
Created on Jul 11, 2016

@author: lxh5147
'''
import unittest
from keras.layers import Input, Lambda
from attention_layer import MaskSequence,mask_sequence, build_bi_directional_layer, remove_mask, add_mask, Attention,ManyToOnePooling, get, get_mask_of_up_level, merge_mask, apply_mask, get_mask, SequenceToSequenceEncoder, SequenceToVectorEncoder, shape, HierarchicalAttention, MLPClassifierLayer, ClassifierWithHierarchicalAttention
from keras import backend as K
import numpy as np
from attention_exp import  faked_dataset


def run(outputs, feed_dict):
    func = K.function(feed_dict.keys(), outputs)
    return func(feed_dict.values())

class AttentionLayerTest(unittest.TestCase):
    def test_get_mask(self):
        x = Input((None,),dtype='int32')
        mask = get_mask(x,padding_id=1)
        f = K.function(inputs=[x],outputs=[mask])
        x_val = [[3,2,1],[0,1,1]]
        mask_val = f([x_val])[0]
        mask_val_ref = [[1,1,0],[1,0,0]]
        self.assertTrue(np.array_equal(mask_val,mask_val_ref),"mask")

    def test_apply_mask(self):
        x = Input((None,2))
        mask = Input((None,), dtype='int32')
        masked_x, normalized_mask = apply_mask(x, mask )

        f=K.function(inputs=[x,mask],outputs=[masked_x,normalized_mask])
        x_val = [[[1,2],[2,3],[3,4]],[[5,6],[1,2],[2,2]]]
        mask_val = [[1,1,0],[1,0,0]]

        masked_x_val_ref=[[[1,2],[2,3],[0,0]],[[5,6],[0,0],[0,0]]]
        normalized_mask_val_ref= [[[1],[1],[0]],[[1],[0],[0]]]
        masked_x_val, normalized_mask_val = f([x_val,mask_val])

        self.assertTrue(np.array_equal(masked_x_val, masked_x_val_ref), "masked_x_val")
        self.assertTrue(np.array_equal(normalized_mask_val, normalized_mask_val_ref), "normalized_mask_val")

    def test_merge_mask(self):
        mask = Input((None,), dtype='int32')
        mask2 = Input((None,), dtype='int32')
        merged_mask = merge_mask(mask,mask2)
        f = K.function(inputs=[mask,mask2], outputs=[merged_mask])
        mask_val = [[1, 1, 0], [1, 0, 0]]
        mask2_val = [[1, 0, 1], [0, 0, 1]]
        merged_mask_val_ref = [[1, 1, 1], [1, 0, 1]]
        merged_mask_val = f([mask_val,mask2_val])[0]
        self.assertTrue(np.array_equal(merged_mask_val, merged_mask_val_ref), "merged_mask_val")

    def test_get_mask_of_up_level(self):
        mask = Input((None,None), dtype='int32')
        updated_mask = get_mask_of_up_level(mask)
        f = K.function(inputs=[mask], outputs=[updated_mask])
        mask_val = [[[1,0], [1,0], [0,0]], [[1,0], [1,0], [0,0]]]
        updated_mask_val_ref = [[1, 1, 0], [1, 1, 0]]
        updated_mask_val = f([mask_val])[0]
        self.assertTrue(np.array_equal(updated_mask_val, updated_mask_val_ref), "updated_mask_val")

    def test_get(self):
        max = get('max')
        self.assertTrue(max == K.max)

    def test_MaskSequence(self):
        layer = ManyToOnePooling(mode='min')
        x=Input((None,None))
        masking= MaskSequence(is_vector_sequence= True)
        masked_x = masking(x)
        output = layer(masked_x) # will call with mask
        f = K.function(inputs=[x],outputs=[output])
        x_val = [[[1, 2], [2, 3], [0, 0]], [[5, 6], [0, 0], [0, 0]]]
        output_val_ref = [[1, 2], [5, 6]]
        output_val = f([x_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_ref), "output_val")

    def test_remove_mask(self):
        x = Input((None, None))
        masking = MaskSequence(is_vector_sequence=True)
        masked_x = masking(x)
        x_without_mask = remove_mask(masked_x)
        # lambda layer does not support mask
        Lambda(lambda input: input)(x_without_mask)
        f = K.function(inputs=[x], outputs=[x_without_mask])
        x_val = [[[1,2],[3,4],[0,0]]]
        output_val = f([x_val])[0]
        self.assertTrue(np.array_equal(output_val, x_val), "output_val")

    def test_add_mask(self):
        x = Input((None, None))
        mask = Input((None,))
        x_with_mask_attached = add_mask(x,mask)
        try:
            Lambda(lambda input: input)(x_with_mask_attached)
        except:
            self.assertTrue(True,"with mask")

    # mask testing
    def test_ManyToOnePooling_mask(self):
        layer = ManyToOnePooling(mode='max')
        x=Input((None,None))
        mask = Input((None,), dtype='int32')
        output = layer.call(x,mask)
        f = K.function(inputs=[x,mask],outputs=[output])
        x_val = [[[1, 2], [2, 3], [3, 4]], [[5, 6], [1, 2], [2, 2]]]
        mask_val = [[1, 1, 0], [1, 0, 0]]
        output_val_ref = [[2, 3], [5, 6]]
        output_val = f([x_val,mask_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_ref), "output_val")

        mask_val = [[1, 1, 1], [0, 1, 1]]
        output_val_ref = [[3, 4], [2, 2]]
        output_val = f([x_val, mask_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_ref), "output_val")


    def test_build_bi_directional_layer(self):
        h1 = Input((None,2))
        h1 = mask_sequence(h1, mask_value=0,is_vector_sequence=True)
        h2 = Input((None, 2))
        h2 = mask_sequence(h2, mask_value=0,is_vector_sequence=True)
        output = build_bi_directional_layer(h1,h2)
        f = K.function(inputs=[h1,h2], outputs=[output])
        h1_val = [[[0,1],[1,2],[0,0]]]
        h2_val = [[[3, 4], [5, 6], [7, 8]]]
        output_val_ref=[[[0,1,7,8], [1,2,5, 6], [0,0,3,4]]]
        output_val = f([h1_val,h2_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_ref), "output_val")


    def test_attention(self):
        attention = Attention(attention_weight_vector_dim=4)
        x = Input(shape=(None, 3))
        mask_x = mask_sequence(x, mask_value=0, is_vector_sequence=True)
        y = attention(mask_x)
        self.assertEqual(shape(y), (None, 3), "y")
        f = K.function(inputs=[x], outputs=[y])
        x_val = [[[1,2,3],[4,5,6],[0,0,0]],[[7,8,9],[0,0,0],[0,0,0]]]
        output_val = f([x_val])[0]
        x_val_2= [[[1,2,3],[4,5,6]],[[7,8,9],[0,0,0]]]
        output_val_2 = f([x_val_2])[0]
        self.assertTrue(np.array_equal(output_val, output_val_2), "output_val")

    def test_transform_sequence_to_sequence(self):
        tensor_input = Input(shape=(2, 3))
        mask = Input((2,), dtype='int32')
        masked_input = add_mask(tensor_input, mask)
        output_dim = 4
        sequence_to_sequence_encoder = SequenceToSequenceEncoder(output_dim)
        output_sequence = sequence_to_sequence_encoder(masked_input)
        self.assertEqual(shape(output_sequence), (None, 2, output_dim*2), "output_sequence")
        f=K.function(inputs=[tensor_input,mask], outputs=[output_sequence])
        x_val = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
        mask_val = [[1,1],[1,0]]
        output_val = f([x_val,mask_val])[0]
        x_val_2 = [[[1, 2, 3], [4,5,6]], [[7, 8, 9], [0, 0, 0]]]
        output_val_2 = f([x_val_2,mask_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_2), "output_val")


    def test_transform_sequence_to_vector_encoder(self):
        output_dim = 2
        sequence_to_vector_encoder = SequenceToVectorEncoder(output_dim)
        tensor_input = Input(shape=(2,3))
        mask = Input((2,), dtype='int32')
        masked_input = add_mask(tensor_input, mask)
        output_vector = sequence_to_vector_encoder(masked_input)
        self.assertEqual(shape(output_vector), (None, output_dim), "output_vector")
        f=K.function(inputs=[tensor_input,mask], outputs=[output_vector])
        x_val = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
        mask_val = [[1,1],[1,0]]
        output_val = f([x_val,mask_val])[0]
        x_val_2 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 0, 0]]]
        output_val_2 = f([x_val_2, mask_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_2), "output_val")


    def test_build_hierarchical_attention_layer_inputs(self):
        # time_steps* documents * sections* sentences * words
        input_shape = (7, 8, 5, 6, 9)
        # record, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)

        self.assertEqual(len(inputs), len(input_feature_dims) + 1, "inputs")
        self.assertEqual(shape(inputs[0]), (None, 7, 8, 5, 6, 9), "inputs")  # original input
        self.assertEqual(shape(inputs[1]), (None, 7, 8, 5, 6, 9, 30), "inputs")  # word features
        self.assertEqual(shape(inputs[2]), (None, 7, 8, 5, 6, 60), "inputs")  # sentence features
        self.assertEqual(shape(inputs[3]), (None, 7, 8, 5, 50), "inputs")  # section features
        self.assertEqual(shape(inputs[4]), (None, 7, 8, 10), "inputs")  # document features
        self.assertEqual(shape(inputs[5]), (None, 7, 20), "inputs")  # snapshot features


    def test_hierarchical_attention_layer(self):
        # snapshots* documents * sections* sentences * words
        input_shape = (7, 8, 5, 6, 9)
        # snapshot, document,section,sentence,word
        input_feature_dims = (20, 10, 50, 60, 30)
        # document, section, sentence, word
        attention_output_dims = (45, 35, 25, 65)
        # document, section, sentence, word
        attention_weight_vector_dims = (82, 72, 62, 52)
        # embedding
        embedding_rows = 1024
        embedding_dim = 50
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        hierarchical_attention = HierarchicalAttention(input_shape, input_feature_dims[0], attention_output_dims,
                                                       attention_weight_vector_dims, embedding_rows, embedding_dim,
                                                       initial_embedding, use_sequence_to_vector_encoder=False)
        output = hierarchical_attention(inputs)
        self.assertEqual(shape(output), (None, 7, 20 + 45 * 2), "output")

    def test_mlp_softmax_classifier(self):
        x = Input(shape=(2, 3))
        mask = Input((2,), dtype='int32')
        masked_input = add_mask(x, mask)
        output_dim = 4
        hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")
        mlp_softmax_classifier = MLPClassifierLayer(output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        y = mlp_softmax_classifier(masked_input)
        self.assertEqual(shape(y), (None, 2, 4), "y")
        output = remove_mask(y)
        f = K.function(inputs=[x, mask], outputs=[output])
        x_val = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        mask_val = [[1, 1], [1, 0]]
        output_val = f([x_val, mask_val])[0]
        x_val_2 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 0, 0]]]
        output_val_2 = f([x_val_2, mask_val])[0]
        self.assertTrue(np.array_equal(output_val, output_val_2), "output_val")

    def test_hierarchical_attention_layer(self):
        input_shape = (3, 2, 4, 2, 6)
        # record, document,section,sentence,word
        input_feature_dims = (2, 3, 4, 5, 6)
        # document, section, sentence, word
        attention_output_dims = (3, 4, 5, 8)
        # document, section, sentence, word
        attention_weight_vector_dims = (3, 4, 5, 6)
        # embedding
        embedding_rows = 10
        embedding_dim = 2
        # classifier
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        hierarchical_attention = HierarchicalAttention(input_shape,input_feature_dims[0], attention_output_dims, attention_weight_vector_dims, embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder = False)
        output = hierarchical_attention(inputs)
        self.assertEqual(shape(output), (None, 3, 8), "y")

        total = 2
        output_dim = 10
        timesteps = input_shape[0]
        x_train, _ = faked_dataset(inputs, total, timesteps, embedding_rows, output_dim)

        # build feed_dic
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[inputs[i]] = x_train[i]
        feed_dict[K.learning_phase()] = 1
        y_out = run(output, feed_dict = feed_dict)
        self.assertEquals(y_out.shape , (2, 3, 8), "y_out")

    def test_hierarchical_attention_with_classifier_layer(self):
        input_shape = (3, 2, 4, 2, 6)
        # record, document,section,sentence,word
        input_feature_dims = (2, 3, 4, 5, 6)
        # document, section, sentence, word
        attention_output_dims = (3, 4, 5, 8)
        # document, section, sentence, word
        attention_weight_vector_dims = (3, 4, 5, 6)
        # embedding
        embedding_rows = 10
        embedding_dim = 2
        initial_embedding = np.random.random((embedding_rows, embedding_dim))
        # classifier
        output_dim = 5
        hidden_unit_numbers = (5, 20)  # 5--> first hidden layer, 20 --> second hidden layer
        hidden_unit_activation_functions = ("relu", "relu")

        use_sequence_to_vector_encoder = False
        inputs = HierarchicalAttention.build_inputs(input_shape, input_feature_dims)
        classifier = ClassifierWithHierarchicalAttention(input_shape,input_feature_dims[0], attention_output_dims, attention_weight_vector_dims,
                                                         embedding_rows, embedding_dim, initial_embedding, use_sequence_to_vector_encoder,
                                                         output_dim, hidden_unit_numbers, hidden_unit_activation_functions)
        output = classifier(inputs)
        total = 2
        timesteps = input_shape[0]
        x_train, _ = faked_dataset(inputs, total, timesteps, embedding_rows, output_dim)

        # build feed_dic
        feed_dict = {}
        for i in range(len(inputs)):
            feed_dict[inputs[i]] = x_train[i]
        feed_dict[K.learning_phase()] = 1

        y_out = run(output, feed_dict = feed_dict)
        self.assertEqual(y_out.shape , (total , timesteps, output_dim), "y_out")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
