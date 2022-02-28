"""
This is an implementation of a model-free Q learning agent. Training the agent involves building the Q table which,
after convergence, essentially represents the value of taking an action in any state of the system. This value has
2 components: an immediate reward and a discounted reward. The action policy is an argmax() function on the action values
for the particular state.

UPDATE RULE:
After every experience (interaction with the world) of the agent, the Q table is updated, by the following rule:
Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax_a'(Q[s', a'])])
where
r = R[s, a] is the immediate reward for taking action a in state s.
γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards.
s' is the resulting next state.
argmax_a'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s'.
α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.

DYNA
An algorithm to speed up learning. Especially useful when 'real world' experience is expensive. The algorithm involves
learning a model of the transition function T (to be able to simulate the next state after performing an action) and a
model of the reward function R (to be able to simulate the reward after performing a certain action). T then is the
probability that if we are in state s and take action a, we end up in state s’ and R is our expected reward if we are in
state s and take action a.

In this implementation, the way the probabilistic model T is learned is based on counting the times each action is taken
in a certain state (This being matrix Tc) during the 'real-world' interactions. R is just the expected reward when in
state s and execute action a. It is calculated based on r, the immediate reward, and also has a component based on the
old value to which it is combined using the learning rate (similarly to updating the Q value).

"""

import random as rand
import numpy as np


class QLearner(object):
    """
    num_states: The number of states to consider.
    num_actions: The number of actions available.
    alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    """  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        num_states=100,  		  	   		   	 			  		 			 	 	 		 		 	
        num_actions=4,  		  	   		   	 			  		 			 	 	 		 		 	
        alpha=0.2,  		  	   		   	 			  		 			 	 	 		 		 	
        gamma=0.9,
        dyna=100,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """
        self.num_actions = num_actions
        self.num_states = num_states

        self.last_action = -1
        self.last_state = -1
        self.Qtable = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.Tc = np.ones((num_states, num_actions, num_states))
        self.Tc = self.Tc * 0.000000001
        self.T = np.zeros((num_states, num_actions, num_states))
        self.R = np.zeros((num_states, num_actions))
        self.dyna = dyna

    def querysetstate(self, s):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Update the state without updating the Q-table.  		  	   		   	 			  		 			 	 	 		 		 	
        
        Input:
        s: The new state  		  	   		   	 			  		 			 	 	 		 		 	
        
        Output:
        action: The selected action  		  	   		   	 			  		 			 	 	 		 		 	  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.last_state = s
        action = np.argmax(self.Qtable[s, :])
        self.last_action = action

        return action

    def query(self, s_prime, r):  		  	   		   	 			  		 			 	 	 		 		 	
        """
        Update the Q table and return an action.  		  	   		   	 			  		 			 	 	 		 		 	

        Input:
        s_prime: The new state  		  	   		   	 			  		 			 	 	 		 		 	
        r: The immediate reward  		  	   		   	 			  		 			 	 	 		 		 	
        
        Output:  		  	   		   	 			  		 			 	 	 		 		 	
        action: The selected action  		  	   		   	 			  		 			 	 	 		 		 	
        """

        # update Qtable
        self.Qtable[self.last_state, self.last_action] = \
            (1 - self.alpha) * self.Qtable[self.last_state, self.last_action] \
            + self.alpha * (r + self.gamma * self.Qtable[s_prime, np.argmax(self.Qtable[s_prime, :])])

        # DYNA
        if self.dyna > 0:
            # update model
            self.Tc[self.last_state, self.last_action, s_prime] += 1
            self.T[self.last_state, self.last_action, s_prime] = \
                self.Tc[self.last_state, self.last_action, s_prime] / np.sum(self.Tc[self.last_state, self.last_action, :])
            self.R[self.last_state, self.last_action] = (1 - self.alpha) * self.R[self.last_state, self.last_action] \
                                                        + self.alpha * r
            for i in range(1, self.dyna):
                # hallucinate
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)
                s_p = np.argmax(self.T[s, a, :])
                r_p = self.R[s, a]
                # update Qtable
                self.Qtable[s, a] = \
                    (1 - self.alpha) * self.Qtable[s, a] \
                    + self.alpha * (r_p + self.gamma * self.Qtable[s_p, np.argmax(self.Qtable[s_p, :])])

        # query for next action
        action = np.argmax(self.Qtable[s_prime, :])
        self.last_action = action
        self.last_state = s_prime

        return action

