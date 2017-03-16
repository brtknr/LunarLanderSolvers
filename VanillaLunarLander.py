#!/usr/bin/env python
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
from gym.wrappers import Monitor

env = gym.make('LunarLander-v2')
print ('Shape of the observation space is', env.observation_space.shape)

# Whether to record videos or not?
if True:
    env = Monitor(env,'VanillaLunarLander', force=True)

D, = env.observation_space.shape # input dimensionality
    
# hyperparameters
H = D*25 # number of hidden layer neurons
BATCH_SIZE = 2 # every how many episodes to do a param update?
LEARNING_RATE = 1e-4 # feel free to play with this to train faster or more stably.
GAMMA = 0.99 # discount factor for reward

epsilon = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.1

tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to 
#giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.get_variable("B1", shape=[H])

layer1 = tf.nn.elu(tf.matmul(observations,W1) + B1)

W2 = tf.get_variable("W2", shape=[H, H],
           initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.get_variable("B2", shape=[H])
layer2 = tf.nn.elu(tf.matmul(layer1,W2) + B2)


W3 = tf.get_variable("W3", shape=[H, env.action_space.n],
           initializer=tf.contrib.layers.xavier_initializer())
B3 = tf.get_variable("B3", shape=[env.action_space.n])
state_value_layer = tf.matmul(layer2,W3) + B3

# probability = tf.nn.softmax(state_value_layer)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
actions = tf.placeholder(tf.float32,[None,env.action_space.n], name="actions")
advantages = tf.placeholder(tf.float32,name="reward_signal")

action_value_vector = tf.reduce_sum(tf.mul(state_value_layer, actions), reduction_indices=1)

loss = tf.reduce_sum(tf.square(advantages - action_value_vector))



# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
# loglik = tf.log(actions*(actions - probability) + (1 - actions)*(actions + probability))
# loglik = actions*(actions - probability) + (1 - actions)*(actions + probability)
# loglik = actions*probability + (1 - actions)*(1 - probability)
# loss = -tf.reduce_sum(loglik * advantages)

# loglik = actions*(actions - probability) + (1 - actions)*(actions + probability)
# loglik = tf.square(actions - probability)
# loss = tf.reduce_sum(loglik * advantages)

newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
W3Grad = tf.placeholder(tf.float32,name="batch_grad3")
batchGrad = [W1Grad,W2Grad,W3Grad]
updateGrads = optimizer.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

    # %%time

xs,hs,dlogps,drs,ys = [],[],[],[],[]
running_reward = None
running_loss = None
reward_sum = 0
loss_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset() # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        print (grad.shape)
        gradBuffer[ix] = grad * 0
    
    while episode_number <= total_episodes:
        
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,D])
        
        # Run the policy network and get an action to take. 
        tfvalue = sess.run(state_value_layer,feed_dict={observations: x})
        # action = 0 if np.random.uniform() < tfprob else 1
        if np.random.uniform() > epsilon:
            action = np.argmax(tfvalue)
        else:
            action = np.random.choice(range(env.action_space.n))
        
        xs.append(x) # observation
        y = action # a "fake label"
        ys.append(y)
        
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        
        if done: 
            epsilon = max(epsilon*EPSILON_DECAY,MIN_EPSILON)

            # print( drs)            
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            # epy = np.vstack(ys)
            epy = np.eye(env.action_space.n)[ys]
            epr = np.vstack(drs)
            xs,hs,dlogps,drs,ys = [],[],[],[],[] # reset array memory
            
            # compute the discounted reward backwards through time
            
            discounted_epr = discount_rewards(epr)
            
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            # discounted_epr -= np.mean(discounted_epr)
            # discounted_epr /= np.std(discounted_epr)
            
            # Get the gradient for this episode, and save it in the gradBuffer
            tLoss,tGrad = sess.run(fetches=(loss,newGrads),feed_dict={observations: epx, actions: epy, advantages: discounted_epr})
            if episode_number%500 == 0:
                for item in zip(discounted_epr,epy):
                    print (item)
                
            # Iterating over the layers
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                
            loss_sum += tLoss

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % BATCH_SIZE == 0: 
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0]/BATCH_SIZE,W2Grad:gradBuffer[1]/BATCH_SIZE,W3Grad:gradBuffer[2]/BATCH_SIZE})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running_loss = tLoss if running_loss is None else running_loss * 0.99 + tLoss * 0.01
                print ('%d Episode reward %f.  Running reward %f. Episode loss %f. Running loss %f.' % (episode_number,reward_sum/BATCH_SIZE, running_reward/BATCH_SIZE, loss_sum/BATCH_SIZE, running_loss/BATCH_SIZE))
                
                if reward_sum/BATCH_SIZE > 200: 
                    print ("Task solved in",episode_number,'episodes!')
                    break
                    
                reward_sum = 0
                loss_sum = 0

            observation = env.reset()
        
print (episode_number,'Episodes completed.')