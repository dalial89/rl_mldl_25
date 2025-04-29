"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb

import gym

from env.custom_hopper import *


import os
import cv2
import gym
from env.custom_hopper import *

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Dynamics parameters:', env.get_parameters())

    n_episodes = 500
    save_frames = True  # invece di render

    frames_dir = "/content/hopper_frames"
    os.makedirs(frames_dir, exist_ok=True)

    frame_idx = 0  # contatore globale dei frame

    for episode in range(n_episodes):
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if save_frames:
                frame = env.render(mode='rgb_array')  # molto importante: mode='rgb_array'
                frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # cv2 vuole immagini BGR
                frame_idx += 1

    env.close()

    # Dopo aver raccolto tutti i frame, creiamo un video
    make_video_from_frames(frames_dir, output_path="/content/hopper_video.avi", fps=30)

def make_video_from_frames(frames_dir, output_path, fps=30):
    import glob

    img_array = []
    filenames = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))

    if not filenames:
        print("No frames found!")
        return

    # Leggiamo la dimensione dal primo frame
    img = cv2.imread(filenames[0])
    height, width, layers = img.shape
    size = (width, height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for filename in filenames:
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print(f"Video salvato in {output_path}")

	

if __name__ == '__main__':
	main()
