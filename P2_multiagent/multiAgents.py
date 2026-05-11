# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "QUESTION 1"

        "*** YOUR CODE HERE ***"
        newNumFood = successorGameState.getNumFood()
        currentScore = scoreEvaluationFunction(currentGameState)
        newScore = successorGameState.getScore()
        
        # Food
        FoodL = newFood.asList()
        if FoodL:
            closestFoodDistance = min([manhattanDistance(newPos,food) for food in FoodL])
        else:
            closestFoodDistance = 0

        foodScore = 15 / (closestFoodDistance + 1)
        foodCountScore = 80 / (newNumFood + 1) 

        # Ghost
        closestGhostDistance = min([
            manhattanDistance(newPos, ghost.getPosition())
            for ghost in newGhostStates
        ])

        smallScareTime = min(newScaredTimes)
        if smallScareTime > 0:
            ghostScore = 20 / (closestGhostDistance + 1)    # đuổi ghost bị sợ
        else:
            ghostScore = -15 / (closestGhostDistance + 1)   # tránh những ghost bình thường

        if action == "Stop":
            return -999;

        scoreDifference = newScore - currentScore

        return foodScore + foodCountScore + ghostScore + scoreDifference

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    "QUESTION 2"

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            numAgents = gameState.getNumAgents()

            # Base case
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            actions = gameState.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(gameState), None
            
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:     # Max node: Pacman
                return max(
                    (minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))[0], action)
                    for action in actions
                ) 
            else:   # Min node: ghost
                return min(
                    (minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))[0], action)
                    for action in actions
                )

        score, action = minimax(0, 0, gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "QUESTION 3"
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        maxResult = float('-inf')
        a = float('-inf')
        b = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            currentResult = self.minValue(successor, 0, 1, a, b)
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = action
                a = max(a, currentResult)
        return maxAction
    
    def maxValue(self, gameState, currentDepth, a, b):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(0)
        maxVal = float('-inf')

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxVal = max(maxVal, self.minValue(successor, currentDepth, 1, a, b))
            if maxVal > b:
                return maxVal
            a = max(a, maxVal)
        return maxVal
    
    def minValue(self, gameState, currentDepth, currentAgent, a, b):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(currentAgent)
        minVal = float('inf')
        agents = gameState.getNumAgents()

        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            
            if currentAgent < agents - 1:
                minVal = min(minVal, self.minValue(successor, currentDepth, currentAgent + 1, a, b))
            else:
                minVal = min(minVal, self.maxValue(successor, currentDepth + 1, a, b))

            if minVal < a:
                return minVal
            
            b = min(b, minVal)
        return minVal  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        "QUESTION 4"

        def expectimax(gameState, agentIndex, depth=0):
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            
            if agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1

            numAction = len(legalActionList)

            if agentIndex == self.index:
                value = float('-inf')
            else:
                value = 0

            for legalAction in legalActionList:
                successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                expectedMax = expectimax(successorGameState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        value = expectedMax
                        bestAction = legalAction
                else:
                    value += (1/numAction) * expectedMax
            return value, bestAction
        
        bestScoreActionPair = expectimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove = bestScoreActionPair[1]
        return bestMove


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    "QUESTION 5"

    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    win = 0
    lose = 0
    if currentGameState.isWin():
        win = 10000000
    elif currentGameState.isLose():
        lose = -10000000

    score = currentGameState.getScore()

    # food
    foodScore = 0
    if foodList:
        minDist = min(manhattanDistance(pacmanPosition, f) for f in foodList)
        foodScore += 10 / (minDist + 1)
        foodScore += sum(1 / (manhattanDistance(pacmanPosition, f) + 1) for f in foodList)
        foodScore -= len(foodList) * 20

    # ghost
    ghostScore = 0
    for ghost in ghostStates:
        d = manhattanDistance(pacmanPosition, ghost.getPosition())

        if ghost.scaredTimer > 0:
            if ghost.scaredTimer > d:       # đuổi khi kịp ăn
                ghostScore += 30 / (d + 1)
            ghostScore += ghost.scaredTimer / (d + 1)
        else:
            if d <= 1:
                ghostScore -= 500
            elif d <= 2:
                ghostScore -= 100
            else:
                ghostScore -= 1 / d
                
    # capsules
    capsuleScore = -200 * len(capsuleList)  # khuyến khích ăn capsule 
    if capsuleList:
        minCapDist = min(manhattanDistance(pacmanPosition, c) for c in capsuleList)
        capsuleScore += 3 / (minCapDist + 1)
    
    return score + foodScore + ghostScore + capsuleScore + win + lose
    
# Abbreviation
better = betterEvaluationFunction
