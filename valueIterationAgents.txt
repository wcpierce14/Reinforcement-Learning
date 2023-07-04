# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # get all states of the MDP

       
        # Loop over iterations
        for i in range(self.iterations):
            temp = util.Counter() # make a blank dictionary for this iteration (used bc it's "batch" updating)

            # Loop over states
            for state in states:
                temp[state] = float('-inf')

                # If terminal state, set value to 0
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                    continue

                # Loop over actions for each state to find V(s)
                for action in self.mdp.getPossibleActions(state):
                    tempVal = self.computeQValueFromValues(state, action) # compute Q-Val from (state, action) pair
                    temp[state] = max(temp[state], tempVal) # update q-value if we found a new max
            self.values = temp
    
        # print(vals)




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = R(s) + \gamma * sum_{s'} P(s' | s, a) * max_{a'} Q(s', a')
        # reward = self.mdp.getReward(state, action)
        # print(self.getQValue(state, action))


        futureStates = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0

        for nextState in futureStates:
            successor = nextState[0]
            probability = nextState[1]
            reward = self.mdp.getReward(state, action, successor)
            discount = self.discount
            successorVal = self.values[successor]
            total += (probability * (reward + discount * successorVal))

        # print(total)
        return total

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get actions
        listOfActions = self.mdp.getPossibleActions(state)

        if len(listOfActions) == 0:
            return None
    
        bestVal = float("-inf")
        bestAction = ""
        for action in listOfActions:
            actionVal = self.computeQValueFromValues(state, action)
            if actionVal >= bestVal:
                bestVal = actionVal
                bestAction = action
        # print(bestAction, bestVal)
        return bestAction
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # get all states of the MDP
        numStates = len(states)

        # Loop over iterations
        for i in range(self.iterations):
            index = i % numStates
            state = states[index]
            bestVal = float('-inf')

            # If terminal state, set value to 0
            if self.mdp.isTerminal(state):
                self.values[state] = 0
                continue

            # Loop over actions for each state to find V(s)
            for action in self.mdp.getPossibleActions(state):
                tempVal = self.computeQValueFromValues(state, action) # compute Q-Val from (state, action) pair
                bestVal = max(bestVal, tempVal)
            self.values[state] = bestVal # update q-value if we found a new max


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

