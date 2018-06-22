# --------------------------------------------------------
# Tensorflow GCA-Net
# Licensed under The MIT License [see LICENSE for details]
# Written by Yimeng Li
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nltk.corpus import wordnet as wn
import json
import numpy as np
from model.config import cfg

answer_dict = json.load(open('/home/mm/workspace/VQA-oqv/dataset/activityNet/qa_anno/dict/answer_one_dic.json'))
class Network(object):
    def __init__(self, handle_types, handle_shapes):
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self.handle_types = handle_types
        self.handle_shapes = handle_shapes

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _build_co_atten(self, activate_tensor, object_tensor, shapes, scope):
        # activate_tensor [batch_size, reduce_dim, activate_dim]
        # object_tensor [batch_size, reduce_dim, oobject_dim]

        with tf.variable_scope(scope):
            activate_reduce = shapes[0]
            activate_dim = shapes[1]
            object_reduce = shapes[2]
            object_dim = shapes[3]
            atten_size = shapes[4]

            # activate att step.1
            embedding_activate = tf.reshape(activate_tensor, [-1,activate_dim])
            w_embedding = tf.get_variable('activate_w', [activate_dim, atten_size], initializer=self.initializer, regularizer=self.weights_regularizer)
            # [batch_size*activate_reduce, atten_size]
            embeded_activate = tf.nn.tanh(tf.matmul(embedding_activate, w_embedding))
            w_activate_g1 = tf.get_variable('w_activate_g1', [atten_size, 1], initializer=self.initializer, regularizer=self.weights_regularizer)
            activate_h1 = tf.reshape(tf.matmul(embeded_activate, w_activate_g1), [-1, activate_reduce])
            activate_p1 = tf.nn.softmax(activate_h1, dim=-1)
            activate_p1 = tf.reshape(activate_p1, [-1, 1, activate_reduce])
            # [batch_size, activate_dim]
            feature_activateAtt1 = tf.reshape(tf.matmul(activate_p1, activate_tensor), (-1, activate_dim))

            # obj att step
            embedding_object = tf.reshape(object_tensor, [-1, object_dim])
            w_object = tf.get_variable('object_w', [object_dim, atten_size], initializer=self.initializer, regularizer=self.weights_regularizer)
            # [batch_size*object_reduce, atten_size]
            obj_embedded = tf.matmul(embedding_object, w_object)

            w_activateAtt1_embed = tf.get_variable('w_activateAtt1_embed', [activate_dim, atten_size], initializer=self.initializer, regularizer=self.weights_regularizer)
            activateAtt1_embeded = tf.matmul(feature_activateAtt1, w_activateAtt1_embed)
            # [batch_size, object_reduce, ATTEN_SIZE]
            activateAtt1_broadcast = tf.tile(tf.reshape(activateAtt1_embeded, [-1,1,atten_size]), [1,object_reduce,1])
            # [batch_size*object_reduce, ATTEN_SIZE]
            activateAtt1_broadcast = tf.reshape(activateAtt1_broadcast, [-1,atten_size])
            obj_h = tf.nn.tanh(tf.add(obj_embedded, activateAtt1_broadcast))
            obj_h = tf.reshape(obj_h, [-1, atten_size])
            w_obj_g = tf.get_variable('w_obj_g', [atten_size, 1], initializer=self.initializer, regularizer=self.weights_regularizer)
            # [batch_size, object_reduce]
            obj_p = tf.reshape(tf.matmul(obj_h, w_obj_g), [-1,object_reduce])
            obj_p = tf.nn.softmax(obj_p, dim=-1)
            out_ojb_p= tf.tile(tf.reshape(obj_p, (-1,object_reduce,1)), (1,1,object_dim))
            # [batch_size, object_reduce, object_dim]
            out_feature_objAtt = tf.multiply(out_ojb_p, object_tensor)
            # [batch_size, 1, object_reduce]
            obj_p = tf.reshape(obj_p, [-1, 1, object_reduce])
            # [batch_size, object_dim]
            feature_objAtt = tf.reshape(tf.matmul(obj_p, object_tensor), (-1, object_dim))
            # [batch_size, object_dim]

            # activate att step2
            activate_embed_2 = tf.reshape(activate_tensor, [-1, activate_dim])
            w_activate_embed2 = tf.get_variable('activate_embed2', [activate_dim, atten_size], initializer=self.initializer, regularizer=self.weights_regularizer)
            activate_embeded_2 = tf.nn.tanh(tf.matmul(activate_embed_2, w_activate_embed2))
            # [batch_size, activate_reduce, atten_size]
            activate_embeded_2 = tf.reshape(activate_embeded_2, [-1, activate_reduce, atten_size])

            w_objAtt_fc = tf.get_variable('w_objAtt_fc', [object_dim, atten_size], initializer=self.initializer, regularizer=self.weights_regularizer)
            # [batch, ATTEN_size]
            objAtt_fced = tf.matmul(feature_objAtt, w_objAtt_fc)
            # [batch_size, activate_reduce, atten_size]
            objAtt_broadcast = tf.tile(tf.reshape(objAtt_fced, [-1,1,atten_size]), [1,activate_reduce,1])
            activate_h_2 = tf.nn.tanh(tf.add(activate_embeded_2, objAtt_broadcast))
            activate_h_2 = tf.reshape(activate_h_2, [-1, atten_size])
            w_activate_g_2 = tf.get_variable('w_ques_g_2', [atten_size, 1], initializer=self.initializer, regularizer=self.weights_regularizer)
            # [batch_size, activate_reduce]
            activate_p_2 = tf.reshape(tf.matmul(activate_h_2, w_activate_g_2), [-1, activate_reduce])
            activate_p_2 = tf.nn.softmax(activate_p_2, dim=-1)
            activate_p_2 = tf.reshape(activate_p_2, [-1,1,activate_reduce])
            # [batch_size, activate_dim]
            feature_activateAtt2 = tf.reshape(tf.matmul(activate_p_2, activate_tensor), (-1,activate_dim))
            out_feature_activateAtt2 = tf.tile(tf.reshape(feature_activateAtt2, [-1,1,activate_dim]), (1, object_reduce, 1))

            # [batch_size, object_reduce, activate_dim+object_dim]
            return tf.concat((out_feature_activateAtt2, out_feature_objAtt), axis=-1)

    def _build_lstm(self, input_tensor, lstm_size, scope):
        with tf.variable_scope(scope):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            lstm_out, state = tf.nn.dynamic_rnn(lstm_cell, input_tensor, dtype=tf.float32)
            return lstm_out

    def _build_network(self, is_training=True):
        # select initializers

        qv_co_atten = self._build_co_atten(self._question, self._video_feature , [cfg.QUES_SEQ, cfg.QUES_SIZE, cfg.FRAME_NUMBER, cfg.VIDEO_SIZE, cfg.ATTEN_SIZE], 'qv-co-attention')

        lstm_1_out = self._build_lstm(qv_co_atten, cfg.LSTM_SIZE,'lstm1')

        softmax_input = lstm_1_out[:,-1]
        with tf.variable_scope('final'):

            w_class = tf.get_variable('w_class', [cfg.LSTM_SIZE, cfg.VOCAB_SIZE], initializer=self.initializer, regularizer=self.weights_regularizer)
            softmax_input = tf.matmul(softmax_input, w_class)
            softmax_output = tf.nn.softmax(softmax_input, dim=-1)
            word_pred = tf.reshape(tf.argmax(softmax_output, axis=1, name='word_pred'), (-1,1))
            cross_entropy = tf.losses.log_loss(self._answer, softmax_output)
            self._loss = tf.reduce_mean(cross_entropy)

        self._losses['class_loss'] = self._loss
        self._losses['reg_loss'] = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
        self._losses['all_loss'] = self._losses['class_loss']+ 0.1*self._losses['reg_loss']
        self._event_summaries.update(self._losses)
        answer_word = tf.reshape(tf.argmax(self._answer, axis=1, name='word'), (-1,1))
        self._answer_code = answer_word
        word_accuracy = tf.to_float(tf.equal(word_pred, answer_word))

        # Set learning rate and momentum
        self.lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        #self.optimizer = tf.train.MomentumOptimizer(self.lr, cfg.TRAIN.MOMENTUM)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # Compute the gradients with regard to the loss
        gvs = self.optimizer.compute_gradients(self._losses['all_loss'])
        self.gvs = gvs
        self.train_op = self.optimizer.apply_gradients(gvs)


        #[ batch_size, 1001]
        self._predictions['word_score'] = softmax_output
        self._predictions['word_pred'] = word_pred
        self._predictions['word_accuracy'] = word_accuracy

        self._score_summaries.update(self._predictions)


        return word_pred


    def create_architecture(self, mode, tag=None):
        self.initializer = tf.random_uniform_initializer(-0.01, 0.01)
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        self.handle = tf.placeholder(tf.string, [])
        data_iterator = tf.data.Iterator.from_string_handle(self.handle, self.handle_types, self.handle_shapes)
        features = data_iterator.get_next()
        _vgg_feature = tf.reshape((features['video_feature'][:,:,0,:4096]), [-1,20,4096])
        _c3d_feature = tf.reshape((features['video_feature'][:,:,1,:4096]), [-1,20,4096])
        self._video_feature = tf.add(0.5*_vgg_feature, 0.5*_c3d_feature)
        self._obj_feature = features['video_feature'][:,:,2:,:4096]
        _question = tf.cast(features['question'], tf.float32)
        with tf.variable_scope('question_embedding'):
            w = tf.get_variable('q_embed', (1001, cfg.QUES_SIZE), initializer=self.initializer, regularizer=self.weights_regularizer)
            _question = tf.reshape(_question, (-1, 1001))
            self._question = tf.reshape(tf.matmul(_question, w), (-1, cfg.QUES_SEQ, cfg.QUES_SIZE))
        self._answer = tf.cast(features['answer'], tf.float32)
        self.qtype = tf.cast(features['qtype'], tf.int32)

        #batch_mean_answer = tf.reshape(tf.reduce_mean(self._answer, axis=0), (1,1001))
        #answer_code = tf.argmax(self._answer, axis=-1)
        # [batch_size, 1]
        #no_label_index = tf.reshape(tf.cast(tf.equal(answer_code, 0), tf.float32), (-1,1))
        #add_answer = tf.matmul(no_label_index, batch_mean_answer)
        #self.loss_answer = self._answer

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        #regularizers

        word_pred = self._build_network(training)
        layers_to_output = {'word_pred':word_pred}
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_summary(self, sess, handle):
        feed_dict = {self.handle: handle}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def test_step(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy, loss = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self._loss,
        ], feed_dict=feed_dict)

        qtype= qtype[0][0]
        word_pred_code= word_pred_code[0][0]
        word_pred_accuracy= word_pred_accuracy[0][0]
        answer_code = answer_code[0][0]

        if answer_code == word_pred_code:
            return [qtype, 1, 1, loss]
        if answer_code == 0:
            return [qtype, word_pred_accuracy, -1, loss]
        else:
            answer = wn.synsets(answer_dict[str(answer_code)][0])
            if len(answer) == 0:
                return [qtype, word_pred_accuracy, -1, loss]
            answer = answer[0]
            if word_pred_code == 0:
                return [qtype, word_pred_accuracy, 0, loss]
            else:
                predic = wn.synsets(answer_dict[str(word_pred_code)][0])
                if len(predic) != 0:
                    predic = predic[0]
                    wup_value = answer.wup_similarity(predic)
                    if wup_value:
                        return [qtype, word_pred_accuracy, wup_value, loss]
                    else:
                        return [qtype, word_pred_accuracy, 0, loss]
                else:
                    return [qtype, word_pred_accuracy, 0, loss]

    def train_step(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy, gvs, loss, _ = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self.gvs,
            self._loss,
            self.train_op
        ], feed_dict=feed_dict)
        qtype= np.reshape(qtype, [-1])
        word_pred_code= np.reshape(word_pred_code, [-1])
        word_pred_accuracy= np.reshape(word_pred_accuracy, [-1])
        answer_code = np.reshape(answer_code, [-1])

        accuracy = np.zeros((5,4), dtype='float')
        for i in range(len(qtype)):
            accuracy[qtype[i]][0] += 1
            accuracy[qtype[i]][1] += word_pred_accuracy[i]
            if answer_code[i] == 0:
                if word_pred_code[i] == 0:
                    accuracy[qtype[i]][2]+=1
                    accuracy[qtype[i]][3]+=1
                continue
            else:
                answer = wn.synsets(answer_dict[str(answer_code[i])][0])
                if len(answer) == 0:
                    continue
                answer = answer[0]
                accuracy[qtype[i]][2] += 1
                if word_pred_code[i] != 0:
                    predic = wn.synsets(answer_dict[str(word_pred_code[i])][0])
                    if len(predic) != 0:
                        predic = predic[0]
                        wup_value = answer.wup_similarity(predic)
                        if wup_value:
                            accuracy[qtype[i]][3] += answer.wup_similarity(predic)

        return accuracy, loss

    def train_step_with_summary(self, sess, handle):
        feed_dict = {self.handle:handle}
        qtype, word_pred_code, answer_code, word_pred_accuracy ,loss, summary, _ = sess.run([
            self.qtype,
            self._predictions['word_pred'],
            self._answer_code,
            self._predictions['word_accuracy'],
            self._loss,
            self._summary_op,
            self.train_op
        ], feed_dict=feed_dict)

        qtype= np.reshape(qtype, [-1])
        word_pred_code= np.reshape(word_pred_code, [-1])
        word_pred_accuracy= np.reshape(word_pred_accuracy, [-1])
        answer_code = np.reshape(answer_code, [-1])


        accuracy = np.zeros((5,4))
        for i in range(len(qtype)):
            accuracy[qtype[i]][0] += 1
            accuracy[qtype[i]][1] += word_pred_accuracy[i]
            if answer_code[i] == 0:
                if word_pred_code[i] == 0:
                    accuracy[qtype[i]][2]+=1
                    accuracy[qtype[i]][3]+=1
                continue
            else:
                answer = wn.synsets(answer_dict[str(answer_code[i])][0])
                if len(answer) == 0:
                    continue
                answer = answer[0]
                accuracy[qtype[i]][2] += 1
                if word_pred_code[i] != 0:
                    predic = wn.synsets(answer_dict[str(word_pred_code[i])][0])
                    if len(predic) != 0:
                        predic = predic[0]
                        wup_value = answer.wup_similarity(predic)
                        if wup_value:
                            accuracy[qtype[i]][3] += answer.wup_similarity(predic)

        return accuracy, loss, summary


