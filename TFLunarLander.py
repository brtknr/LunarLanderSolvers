import gym
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

# Summary
run_params = 'truncated_normal_8x92x64x4_012_'
writer = tf.summary.FileWriter('TFLunarLander/run_' + run_params + str(time.time()))

# Action choosing policy
epsilon = .54
epsilon_decay = .984

# Set learning parameters
lr = .99
num_episodes = 1000


def build_graph(input_size=8,
                output_size=4,
                hidden_layers=None,
                learning_rate=0.012):
    x = tf.placeholder(tf.float32, [None, input_size], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, output_size], name='output_placeholder')

    if hidden_layers is None:
        hidden_layers = []
    sizes = [input_size] + hidden_layers
    hidden = x
    layer_idx = 0
    for input, output in zip(sizes[:-1], sizes[1:]):
        layer_idx += 1
        with tf.variable_scope('hidden' + str(layer_idx)):
            W = tf.get_variable('W', [input, output], initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('b', [output], initializer=tf.constant_initializer(0.0))
            hidden = tf.matmul(hidden, W) + b
            hidden = tf.nn.relu(hidden)

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", hidden)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [sizes[-1], output_size], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(hidden, W) + b

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)

    predictions = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        total_loss = tf.reduce_mean(tf.square(y - logits))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    tf.summary.scalar("total_loss", total_loss)

    summary = tf.summary.merge_all()

    return dict(
        x=x,
        y=y,
        total_loss=total_loss,
        train_step=train_step,
        rewards=logits,
        preds=predictions,
        saver=tf.train.Saver(),
        summary=summary
    )


def multinomial(chances):
    x = random.random()
    for idx, y in enumerate(chances):
        if x <= y:
            # print(chances, idx)
            return idx
        else:
            x -= y


env = gym.make('LunarLander-v2')

# Replay Memory
replay_memory = []
with tf.Session() as session:
    g = build_graph(hidden_layers=[92, 64])
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    for i in range(num_episodes):
        epsilon *= epsilon_decay
        # Reset environment and get first new observation
        state = env.reset()
        done = False
        total_score = 0

        # Play
        step = 0
        while step < num_episodes - 1 and not done:
            step += 1
            old_state = state

            # Select an action a, without epsilon for now
            predictions = session.run([g['preds']], feed_dict={g['x']: [state]})
            # Carry out action a with a multinomial probability and observe reward and new state
            action = multinomial(predictions[0][0])
            if random.random() < epsilon:
                action = random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            if i%10 == 0:
                env.render()
            total_score += reward
            if done:
                state = None

            # Store replay memory
            replay_memory.append((old_state, action, reward, state))

        # Learn
        step = 0
        relevant_memory = random.sample(replay_memory, min(len(replay_memory), 100))
        batch_inputs = []
        batch_outputs = []
        for relevant_memory in relevant_memory:
            old_state = relevant_memory[0]
            action = relevant_memory[1]
            reward = relevant_memory[2]
            state = relevant_memory[3]

            outputs = session.run([g['rewards']], feed_dict={g['x']: [old_state]})[0][0]
            observed_reward = reward
            if state is not None:
                future_rewards = session.run([g['rewards']], feed_dict={g['x']: [state]})
                expected_reward = np.amax(future_rewards[0][0])
                observed_reward = reward + lr * expected_reward

            outputs[action] = observed_reward

            batch_inputs.append(old_state)
            batch_outputs.append(outputs)

        replay_memory = replay_memory[int(len(replay_memory) / 10):]

        train_loss, summary, _ = session.run([g['total_loss'], g['summary'], g['train_step']],
                                             feed_dict={g['x']: batch_inputs, g['y']: batch_outputs})

        writer.add_summary(summary, i)

        value = summary_pb2.Summary.Value(tag="Score", simple_value=total_score)
        score_summary = summary_pb2.Summary(value=[value])
        writer.add_summary(score_summary, i)

        print('Score / loss at episode %d: %f / %f' % (i, total_score, train_loss))