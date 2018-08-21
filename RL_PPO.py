"""
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np


A_LearningRate = 0.0001
C_LearningRate = 0.0002
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIMENTION, A_DIMENTION = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class RL(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfstate = tf.placeholder(tf.float32, [None, S_DIMENTION], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfstate, 100, tf.nn.relu)
            self.variable = tf.layers.dense(l1, 1)
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advantage = self.discounted_reward - self.variable
            with tf.variable_scope('criticloss'):
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                tf.summary.scalar('criticloss', self.closs) 
                with tf.variable_scope('atrain'):
                    self.ctrain_op = tf.train.AdamOptimizer(C_LearningRate).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfaction = tf.placeholder(tf.float32, [None, A_DIMENTION], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfaction) - oldpi.log_prob(self.tfaction))
                ratio = pi.prob(self.tfaction) / oldpi.prob(self.tfaction)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflambda = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflambda * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
            tf.summary.scalar('actorloss', self.aloss) 

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LearningRate).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, reward):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfstate: state, self.discounted_reward: reward})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfstate: state, self.tfaction: action, self.tfadv: adv, self.tflambda: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfstate: state, self.tfaction: action, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfstate: state, self.discounted_reward: reward}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfstate, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIMENTION, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIMENTION, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.tfstate: state})[0]
        return np.clip(action, -2, 2)

    def get_variable(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.variable, {self.tfstate: state})[0, 0]