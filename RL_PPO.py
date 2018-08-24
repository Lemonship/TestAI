"""
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
from NetworkFactory import NetworkFactory as ns


class RL(object):

    def __init__(
        self,
        A_LearningRate = 0.0001,
        C_LearningRate = 0.0002,
        A_UpdateStep = 10,
        C_UpdateStep = 10,
        S_Dimension = 3, 
        A_Dimension = 1,
        epsilon=0.2,
        complexity=100
        ):
        self.A_LearningRate = A_LearningRate
        self.C_LearningRate = C_LearningRate
        self.A_UpdateStep = A_UpdateStep 
        self.C_UpdateStep = C_UpdateStep 
        self.S_Dimension = S_Dimension 
        self.A_Dimension = A_Dimension
        self.epsilon = epsilon
        self.complexity = complexity

        self.sess = tf.Session()
        self.tfstate = tf.placeholder(tf.float32, [None, self.S_Dimension], 'State')

        # Critic
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(self.tfstate, self.complexity, tf.nn.relu, name='HiddenLayer')
            self.value = tf.layers.dense(l1, 1,name='Value')
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'DiscountedReward')
            self.advantage = self.discounted_reward - self.value
            with tf.variable_scope('CriticLoss'):
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                tf.summary.scalar('CriticLoss', self.closs) 
        with tf.variable_scope('CriticTrain'):
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LearningRate).minimize(self.closs)

        # Actor
        with tf.variable_scope('Actor'):
            # pi, pi_params = self.build_norm_dist_network('pi', trainable=True)
            pi, pi_params = ns.NormDistFactory(self.tfstate, self.A_Dimension, self.complexity, 'pi', trainable=True)
            oldpi, oldpi_params = ns.NormDistFactory(self.tfstate, self.A_Dimension, self.complexity, 'oldpi', trainable=False)
            with tf.variable_scope('SampleAction'):
                self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
            with tf.variable_scope('UpdateOldpi'):  #Copy Param from pi to oldpi
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            self.tfaction = tf.placeholder(tf.float32, [None, self.A_Dimension], 'Action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'Advantage')
            with tf.variable_scope('ActorLoss'):
                #Control the rate of chanage of action
                with tf.variable_scope('Surrogate'):
                    # ratio = tf.exp(pi.log_prob(self.tfaction) - oldpi.log_prob(self.tfaction))
                    ratio = pi.prob(self.tfaction) / oldpi.prob(self.tfaction)
                    surr = ratio * self.tfadv
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*self.tfadv))
                tf.summary.scalar('ActorLoss', self.aloss) 

        with tf.variable_scope('ActorTrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LearningRate).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, reward):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfstate: state, self.discounted_reward: reward})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        [self.sess.run(self.atrain_op, {self.tfstate: state, self.tfaction: action, self.tfadv: adv}) for _ in range(self.A_UpdateStep)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfstate: state, self.discounted_reward: reward}) for _ in range(self.C_UpdateStep)]

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.tfstate: state})[0]
        return np.clip(action, -2, 2)

    def get_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.value, {self.tfstate: state})[0, 0]