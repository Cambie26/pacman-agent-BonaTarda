# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

#from pyexpat import features
from pyexpat import features
import random
#from turtle import width
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveReflexAgent', second='OffensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """


    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)

        self.turn_counter = 0 # to keep track of how long we've been playing, can be used for timing-based strategies
        #self.map_mid_y = game_state.data.layout.height // 2
        
        # identify crossing points to get back to safety
        layout = game_state.data.layout
        width = layout.width
        height = layout.height

        if self.red:
            self.border_x = width // 2 - 1
        else:
            self.border_x = width // 2

        self.boundary = []
        for y in range(height):
            if not layout.is_wall((self.border_x, y)):
                self.boundary.append((self.border_x, y))

        # assigns one agent to start at top and one at bottom to spread out 
        team = self.get_team(game_state)
        self.prefer_top = (self.index == min(team))# assigns first agent to top half and second to bottom half
        top_target = (self.border_x, int(height * 0.9))
        bottom_target = (self.border_x, int(height * 0.1))
        self.opening_target = top_target if self.prefer_top else bottom_target


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """


        self.turn_counter += 1
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
         

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):

    # added code to get quickest path back to safety by precalculating border points 
    # code to get power capsule and then eat ghoast 
    # more pellets it carries the more likely it is to return home 
    # code to always keep moving
    # fined tuned weights for performance
    # code to return home as end game approaches


    def get_features(self, game_state, action):
        features = util.Counter()
        
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()


        my_y = int(my_pos[1])
        # opening bias to encourage agents to spread out and cover more ground, with a preference for the top or bottom half of the map depending on the agent index
        if self.turn_counter < 31: # only apply opening bias for first 30 turns (or 120 timer steps)
            dist = self.get_maze_distance(my_pos, self.opening_target)
            #features['opening_bias'] = dist


        # eat pellet if very close 
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # distance to visible ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        
        # identify capsules on enemy side
        current_capsules = [
            c for c in game_state.get_capsules()
            if (self.red and c[0] > self.border_x) or (not self.red and c[0] < self.border_x)
        ]

        next_capsules = [
            c for c in successor.get_capsules()
            if (self.red and c[0] > self.border_x) or (not self.red and c[0] < self.border_x)
        ]
        # reward actually eating a capsule
        if len(next_capsules) < len(current_capsules):
            features['eat_capsule'] = 1

        # attraction toward nearest remaining capsule
        if next_capsules:
            nearest_capsule = min(next_capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            min_capsule_dist = self.get_maze_distance(my_pos, nearest_capsule)

            if len(ghosts) > 0:
                scared_times = [g.scared_timer for g in ghosts]
                min_scared = min(scared_times)

                # ghosts dangerous: strong capsule incentive
                if min_scared == 0:
                    features['power_capsule'] = min_capsule_dist

                # ghosts about to recover: mild incentive
                elif min_scared <= 2:
                    features['power_capsule'] = 0.5 * min_capsule_dist

                # optional: only go if we can plausibly beat ghosts to it
                ghost_capsule_dist = min(
                    self.get_maze_distance(g.get_position(), nearest_capsule)
                    for g in ghosts
                )
                if min_capsule_dist >= ghost_capsule_dist:
                    features['power_capsule'] = 0

            else:
                # no visible ghosts: weak attraction only
                features['power_capsule'] = 0.3 * min_capsule_dist

        if len(ghosts) > 0:
            
            ghost_states = ghosts
            ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in ghost_states]
            closest_ghost = min(ghost_distances)

            # check if ghosts are scared
            scared_times = [g.scared_timer for g in ghost_states]
            min_scared = min(scared_times)

            if (min_scared > 3) and (min_scared > closest_ghost + 1):   # enough time left to chase safely
                features['chase_ghost'] = -(6 - closest_ghost) ** 2 - 2
                

            else:   # ghost dangerous
                features['ghost_distance'] = closest_ghost

                if closest_ghost <= 3:
                    features['ghost_danger'] = 4 - closest_ghost


        min_dist_to_safety = min(self.get_maze_distance(my_pos, b) for b in self.boundary)
        if my_state.num_carrying > 0:

            features['return_home'] = min_dist_to_safety * my_state.num_carrying

            moves_left = 300 - self.turn_counter
            # 🚨 panic if we might not make it back
            if moves_left < min_dist_to_safety + 5:
                features['return_home'] *= 10   # scale, don't overwrite

            

        if action == Directions.STOP:
            features['stop'] = 1
        

        
        return features


    def get_weights(self, game_state, action):
        # notes: ghost danger should be strongest for survival
        # 
        return {
            'opening_bias': -400, # encourage spreading out just at the start
            'successor_score': 100, # if pellet nearby
            'distance_to_food': -1.5, # attraction to pellet
            'power_capsule': -30, # attraction to power capsule (if no ghost)
            'eat_capsule': 500, # strong reward for actually eating capsule
            'chase_ghost': -30, # (if power capsule active) attraction to ghost at distance 5 (increases nonlinearly the closer it gets to ensure final commitment)
            'ghost_distance': 2.1, #if ghost is within 5
            'ghost_danger': -80, #if ghost is within 3 (negative cos ghoast distance is inverted)
            'return_home': -1, #return attraction per pellet carrying
            'stop': -1000 # always keep moving
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.

    """
    # Added code to run away when scared.

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            if successor.get_agent_state(self.index).scared_timer == 0:
                features['invader_distance'] = min(dists)
            else:
                features['invader_distance'] = -min(dists) 

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000, 
            'on_defense': 100, 
            'invader_distance': -30, 
            'stop': -1000, 
            'reverse': -2
            }
