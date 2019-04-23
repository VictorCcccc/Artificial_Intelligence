import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.points = 0
        self.s = None
        self.a = None

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''


        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2].copy()
        food_x = state[3]
        food_y = state[4]

        adjoining_wall_x = 0
        adjoining_wall_y = 0
        food_dir_x = 0
        food_dir_y = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        if snake_head_x == 40:
            adjoining_wall_x = 1
        elif snake_head_x == 480:
            adjoining_wall_x = 2

        if snake_head_y == 40:
            adjoining_wall_y = 1
        elif snake_head_y == 480:
            adjoining_wall_y = 2

        if food_x < snake_head_x:
            food_dir_x = 1
        elif food_x > snake_head_x:
            food_dir_x = 2

        if food_y < snake_head_y:
            food_dir_y = 1
        elif food_y > snake_head_y:
            food_dir_y = 2

        if (snake_head_x, snake_head_y - 40) in snake_body:
            adjoining_body_top = 1

        if (snake_head_x, snake_head_y + 40) in snake_body:
            adjoining_body_bottom = 1

        if (snake_head_x - 40, snake_head_y) in snake_body:
            adjoining_body_left = 1

        if (snake_head_x + 40, snake_head_y) in snake_body:
            adjoining_body_right = 1

        discretized_list = [adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]

        # training
        if self._train:
            if self.s != None and self.a != None:
                # update Q table
                s_copy = list(self.s)
                s_copy.append(self.a)
                Q_key = tuple(s_copy)
                Q_s_a_old = self.Q[Q_key].copy()

                alpha = self.C / (self.C + self.N[self.s][self.a])

                R_s = -0.1
                if self.points < points:
                    R_s = 1.0
                elif dead:
                    R_s = -1.0
                self.points = points

                s_prime = discretized_list.copy()
                Q_sp_ap_list = self.Q[tuple(s_prime)]
                Q_sp_ap_list_reverse = list(Q_sp_ap_list)
                Q_sp_ap_list_reverse.reverse()
                max_Q_sp_ap = max(Q_sp_ap_list_reverse)

                self.Q[Q_key] = Q_s_a_old + alpha * (R_s + self.gamma * max_Q_sp_ap - Q_s_a_old)

                # if dead:
                #     self.reset()
                #     return None

            if dead:
                self.reset()
                return None

            # get the next action
            Q_s_ap_list = self.Q[tuple(discretized_list)].copy()
            for i in range(4):
                N_key = list(discretized_list)
                N_key.append(self.actions[i])
                if self.N[tuple(N_key)] < self.Ne:
                    Q_s_ap_list[i] = 1
                    
            Q_s_ap_list_reverse = list(Q_s_ap_list)
            Q_s_ap_list_reverse.reverse()
            max_action_index_reverse = Q_s_ap_list_reverse.index(max(Q_s_ap_list_reverse))
            max_action_index = 3 - max_action_index_reverse

            # update N table
            N_update_key = list(discretized_list)
            N_update_key.append(self.actions[max_action_index])
            self.N[tuple(N_update_key)] += 1

            self.s = tuple(discretized_list.copy())
            self.a = self.actions[max_action_index]

            return self.actions[max_action_index]

        # testing
        Q_list = self.Q[tuple(discretized_list)].copy()
        Q_list_reverse = list(Q_list)
        Q_list_reverse.reverse()
        action_index_reverse = Q_list_reverse.index(max(Q_list_reverse))
        action_index = 3 - action_index_reverse

        return self.actions[action_index]