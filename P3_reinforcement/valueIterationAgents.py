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
        for i in range(self.iterations):
            # Tạo "thùng chứa tạm". util.Counter() hoạt động như một Dictionary mặc định giá trị 0
            new_values = util.Counter() 
            
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                    continue
                
                # Tìm max Q-value cho trạng thái này
                best_action = self.computeActionFromValues(state)
                if best_action is not None:
                    new_values[state] = self.computeQValueFromValues(state, best_action)
            
            # Cập nhật hàng loạt (Batch update)
            self.values = new_values

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
        q_value = 0
        # Duyệt qua các trạng thái tiếp theo có thể xảy ra và xác suất của chúng
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            # Công thức toán học
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
            
        best_action = None
        max_q = float('-inf') # Khởi tạo giá trị lớn nhất bằng âm vô cùng
        
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_q:
                max_q = q_value
                best_action = action
                
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        # Bước 1: Tính toán tiền bối (Predecessors) của tất cả các trạng thái
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
            
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[nextState].add(state)

        # Bước 2: Khởi tạo Hàng đợi ưu tiên (Priority Queue)
        pq = util.PriorityQueue()
        
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Tìm giá trị Q lớn nhất (max Q-value) cho trạng thái hiện tại
                max_q = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                # Tính sai số (diff)
                diff = abs(self.values[state] - max_q)
                # Đưa vào hàng đợi với mức độ ưu tiên là -diff (vì đây là min-heap)
                pq.push(state, -diff)

        # Bước 3: Vòng lặp cập nhật ưu tiên
        for i in range(self.iterations):
            if pq.isEmpty():
                break
                
            # Lấy trạng thái có sai số lớn nhất ra khỏi hàng đợi
            state = pq.pop()
            
            # Cập nhật giá trị của trạng thái đó (không dùng thùng chứa tạm như Q1 nữa)
            if not self.mdp.isTerminal(state):
                max_q = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = max_q
                
            # Cập nhật các tiền bối của trạng thái vừa được xử lý
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    max_q_p = max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)])
                    diff_p = abs(self.values[p] - max_q_p)
                    # Nếu sai số của tiền bối lớn hơn ngưỡng theta, cập nhật nó trong hàng đợi
                    if diff_p > self.theta:
                        pq.update(p, -diff_p)
        