"""
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class RL(object):

    def __init__(
        self,
        A_LearningRate = 0.0001,
        C_LearningRate = 0.0002,
        A_UpdateStep = 10,
        C_UpdateStep = 10,
        S_Dimension = 3, 
        A_Dimension = 1
        ):
        self.A_LearningRate = A_LearningRate
        self.C_LearningRate = C_LearningRate
        self.A_UpdateStep = A_UpdateStep 
        self.C_UpdateStep = C_UpdateStep 
        self.S_Dimension = S_Dimension 
        self.A_Dimension = A_Dimension 

        self.sess = tf.Session()
        self.tfstate = tf.placeholder(tf.float32, [None, self.S_Dimension], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfstate, 100, tf.nn.relu)
            self.value = tf.layers.dense(l1, 1)
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advantage = self.discounted_reward - self.value
            with tf.variable_scope('criticloss'):
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                tf.summary.scalar('criticloss', self.closs) 
                with tf.variable_scope('c-train'):
                    self.ctrain_op = tf.train.AdamOptimizer(self.C_LearningRate).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfaction = tf.placeholder(tf.float32, [None, self.A_Dimension], 'action')
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

        with tf.variable_scope('a-train'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LearningRate).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, state, action, reward):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfstate: state, self.discounted_reward: reward})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UpdateStep):
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
            [self.sess.run(self.atrain_op, {self.tfstate: state, self.tfaction: action, self.tfadv: adv}) for _ in range(self.A_UpdateStep)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfstate: state, self.discounted_reward: reward}) for _ in range(self.C_UpdateStep)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfstate, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.A_Dimension, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.A_Dimension, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.tfstate: state})[0]
        return np.clip(action, -2, 2)

    def get_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.value, {self.tfstate: state})[0, 0]