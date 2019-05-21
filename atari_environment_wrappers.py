import cv2
import gym
import gym.spaces
import numpy as np
import collections


# Class that defines method for skipping N number of frames
# the motivation here is that going from 1 frame to an adjacent frame might not reveal much of a difference to our
# network. skipping N frames still gives us proper dynamics while only revealing important information
# can increase training complexity/time
class SkipFrames(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipFrames, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # need to take the maximum of two last frames because of the flickering effect atari games have
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        # clears past frame buffer and init to first observation from environment
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# class for pressing fire so our agent does not need to do this to start the game
# streamlines the process of game initialization and improves training time
class PressFire(gym.Wrapper):
    def __init___(self, env=None):
        super(PressFire, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


# Preprocess the frames to be smaller and be monochromatic. this reduces the dimenstionality of the game observations
# which will reduce training complexity and time.
class ResizeScreenFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ResizeScreenFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ResizeScreenFrame.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            screen_capture = np.reshape(frame, [210, 160, 3])
        else:
            assert False, "Declared Unknown Screen Size"
        gray_screen_capture = cv2.cvtColor(screen_capture, cv2.COLOR_BGR2GRAY)
        resized_screen_capture = cv2.resize(gray_screen_capture, (84, 84), interpolation=cv2.INTER_AREA)
        reshaped_screen_capture = np.reshape(resized_screen_capture, [84, 84, 1])
        return reshaped_screen_capture.astype(np.uint8)


# The dimenstionality from the observation is different from the convolutional layer expected dimenstionality.
class ConvertFrameDimenstionality(gym.ObservationWrapper):
    def __init__(self, env):
        super(ConvertFrameDimenstionality, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# Create Stacked frames to feed to the network so that the momentum of the game state is dynamic
class CreateStackedFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(CreateStackedFrames, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# Converts observation pixel values into floats that easier for the convolutional network to process
class ConvertPixelBytesToFloats(gym.ObservationWrapper):
        def observation(self, obs):
            return np.array(obs).astype(np.float32) / 255.0


# function that takes the basic game environment and applies all the wrappers to it
def generate_game_environment(env_name):
    env = gym.make(env_name)
    env = SkipFrames(env)
    env = PressFire(env)
    env = ResizeScreenFrame(env)
    env = ConvertFrameDimenstionality(env)
    env = CreateStackedFrames(env, 4)
    return ConvertPixelBytesToFloats(env)


