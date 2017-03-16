import argparse
import logging
import sys

import numpy as np
import math

import gym
from gym import wrappers

"""
According to the code : 
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
"""

class PIDAgent(object):

    # previous_error = 0
    # integral = 0 
    # start:
    #   error = setpoint - measured_value
    #   integral = integral + error*dt
    #   derivative = (error - previous_error)/dt
    #   output = Kp*error + Ki*integral + Kd*derivative
    #   previous_error = error
    #   wait(dt)
    #   goto start

    def __init__(self, Kp, Ki, Kd, indexObservation):

        self.error = 0
        self.previous_error = 0
        self.derivative = 0
        self.integral = 0

        self.index = indexObservation

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def act(self, command, observation, reward, done):

        self.previous_error = self.error
        self.error = command-observation[self.index]
        self.derivative = self.error - self.previous_error
        self.integral = self.integral + self.error

        self.output = self.Kp*(self.error + self.Ki*self.integral + self.Kd*self.derivative)
        return self.output



if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'ScriptedLunarLander'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    horizontalAgent = PIDAgent(-1, 0, -.001, 0)
    verticalAgent = PIDAgent(2, 0, 0, 3)
    angularAgent = PIDAgent(4, 0, -.1, 4)

    episode_count = 100

    observation = None
    reward = 0
    done = False
    info = None

    horizontalAction = 0
    verticalAction = 0
    angularAction = 0


    for i in range(episode_count):
        observation = env.reset()
        for t in range(1000):

            # print(observation)

            horizontalAction = horizontalAgent.act(0, observation, reward, done)
            verticalAction   = verticalAgent.act(-.1, observation, reward, done)
            angularAction    = angularAgent.act(horizontalAction, observation, reward, done)

            if observation[1] < 0.05:
                angularAction = 0


            # print(horizontalAction)
            # print(verticalAction)
            # print(angularAction)

            if (observation[6] == 0 and observation[7] == 0) and (observation[2] != 0 or observation[3] != 0):

                if verticalAction > math.fabs(angularAction):
                    observation, reward, done, info = env.step(2)

                elif angularAction > 0 and observation[1] > 0.05:
                    observation, reward, done, info = env.step(1)
                elif angularAction < 0 and observation[1] > 0.05:
                    observation, reward, done, info = env.step(3)

                else:
                    observation, reward, done, info = env.step(0)

            else:
                observation, reward, done, info = env.step(0)


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.


    # Close the env and write monitor result info to disk
    env.close()