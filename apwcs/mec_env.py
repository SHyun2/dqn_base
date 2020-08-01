"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import random

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class MecEnv(gym.Env):
    # """
    # Description:
    #     A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    # Source:
    #     This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    # Observation:
    #     Type: Box(4)
    #     Num	Observation                 Min         Max
    #     0	Cart Position             -4.8            4.8
    #     1	Cart Velocity             -Inf            Inf
    #     2	Pole Angle                 -24 deg        24 deg
    #     3	Pole Velocity At Tip      -Inf            Inf
    #
    # Actions:
    #     Type: Discrete(2)
    #     Num	Action
    #     0	Push cart to the left
    #     1	Push cart to the right
    #
    #     Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    # Reward:
    #     Reward is 1 for every step taken, including the termination step
    # Starting State:
    #     All observations are assigned a uniform random value in [-0.05..0.05]
    # Episode Termination:
    #     Pole Angle is more than 12 degrees
    #     Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
    #     Episode length is greater than 200
    #     Solved Requirements
    #     Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    # """
    """
    Observation:
        Type: Box(11)
        Num     Observation                     Min     Max
        0       UE Capacity                     0       50
        1       ES Capacity                     0       1000

        2       Task                            0       10000

        3       UE Task Done                    0       50
        4       ES Task Done                    0       1000

        E_Local = UE Processing Energy
        E_Total = Upload Energy + ES Processing Energy + Download Energy
        
        T_Local = UE Processing Delay
        T_Total = Upload Delay + ES Processing Delay + Download Delay
        
    Actions:
        Type: Discrete(2)
        Num     Action
        0       No offload
        1       Offload
        
    Rewards:
        Num     Reward                  Annotation      
        0       UE Energy Consumption   R_E             
        1       UE Execution Delay      R_T             
        
        R_E = -(E_Total / E_local)
        R_T = (T_Local - T_Total)
        
    Starting State:
        All observations are assigned a uniform random value.
        
        Observation     Value Range
        UE Cap          [30..50]
        ES Cap          [500..1000]
        Task            [10..10000]
        
    Episode Termination:
        Episode length is greater than 200
        Solved Requirements (Task = 0)
    """

    def __init__(self):
        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        # self.kinematics_integrator = 'euler'
        #
        # # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        #
        # # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        # high = np.array([
        #     self.x_threshold * 2,
        #     np.finfo(np.float32).max,
        #     self.theta_threshold_radians * 2,
        #     np.finfo(np.float32).max])
        #
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #
        # self.seed()
        # self.viewer = None
        # self.state = None
        #
        # self.steps_beyond_done = None

        # Constants
        self.UE_CAP_MAX = 50
        self.ES_CAP_MAX = 1000
        self.TASK_MAX = 10000

        self.ue_proc_e = 0.4
        self.upload_e = 0.1
        self.download_e = 0.1
        self.es_proc_e = 0.01

        self.ue_proc_d = 0.3
        self.upload_d = 0.2
        self.download_d = 0.2
        self.es_proc_d = 0.03

        self.reward_ratio = 0.5

        self.action_space = spaces.Discrete(2)
        low = np.array([0, 0, 0, 0, 0])
        high = np.array([self.UE_CAP_MAX, self.ES_CAP_MAX, self.TASK_MAX, self.UE_CAP_MAX, self.ES_CAP_MAX])
        self.observation_space = spaces.Box(low, high, dtype=np.int)

        self.seed()
        self.state = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # state = self.state
        # x, x_dot, theta, theta_dot = state
        # force = self.force_mag if action == 1 else -self.force_mag
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #         self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # if self.kinematics_integrator == 'euler':
        #     x = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else:  # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot
        # self.state = (x, x_dot, theta, theta_dot)
        # done = x < -self.x_threshold \
        #        or x > self.x_threshold \
        #        or theta < -self.theta_threshold_radians \
        #        or theta > self.theta_threshold_radians
        # done = bool(done)
        #
        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        #     self.steps_beyond_done += 1
        #     reward = 0.0
        #
        # return np.array(self.state), reward, done, {}

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        state = self.state
        ue_cap, es_cap, task, ue_prev_task, es_prev_task = state

        offload = bool
        # Decide to offload or not
        if action == 1:
            offload = True
        elif action == 0:
            offload = False
        else:
            return EnvironmentError

        # Decide how much to do task
        # task_todo = int(self.np_random.uniform(0, task))
        task_todo = random.randint(0, task)

        # Work
        if offload:
            if es_cap < task_todo:
                task_todo = es_cap
        else:
            if ue_cap < task_todo:
                task_todo = ue_cap

        task -= task_todo

        # # Done previous task
        # ue_cap += ue_prev_task
        # es_cap += es_prev_task

        # if offload:
        #     ue_prev_task = 0
        #     es_prev_task = task_todo
        # else:
        #     ue_prev_task = task_todo
        #     es_prev_task = 0

        self.state = ue_cap, es_cap, task, ue_prev_task, es_prev_task

        # Calculate reward
        e_local = self.ue_proc_e * task_todo + 0.000001
        e_total = (self.upload_e + self.es_proc_e + self.download_e) * task_todo

        t_local = self.ue_proc_d * task_todo
        t_total = (self.upload_d + self.es_proc_d + self.download_d) * task_todo

        reward_e = 0.0
        reward_d = 0.0

        if offload:
            reward_e = -(e_total / e_local)
            reward_d = t_local - t_total

        reward = self.reward_ratio * reward_e + (1 - self.reward_ratio) * reward_d

        done = (task == 0)

        return np.array(self.state), reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.steps_beyond_done = None
        # return np.array(self.state)

        new_ue_cap = int(self.np_random.uniform(30, 50))
        new_es_cap = int(self.np_random.uniform(500, 1000))
        new_task = int(self.np_random.uniform(10, 10000))

        self.state = np.array([new_ue_cap, new_es_cap, new_task, 0, 0])

        return self.state

    def render(self, mode='human'):
        # screen_width = 600
        # screen_height = 400
        #
        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        #
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        #     pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     pole.set_color(.8, .6, .4)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(.5, .5, .8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = rendering.Line((0, carty), (screen_width, carty))
        #     self.track.set_color(0, 0, 0)
        #     self.viewer.add_geom(self.track)
        #
        #     self._pole_geom = pole
        #
        # if self.state is None: return None
        #
        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        #
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        pass

    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        pass
