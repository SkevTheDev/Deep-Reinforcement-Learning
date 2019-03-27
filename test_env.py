import gym
import tensorflow as tf

print(tf.__version__)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

env = gym.make('CartPole-v0')
env.reset()
