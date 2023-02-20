# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    visited = set() # A set to keep track of visited states
    stack = util.Stack() # A stack to keep track of nodes to be explored
    stack.push((problem.getStartState(), [], 0)) # Add the start state to the stack with an empty action list and a cost of 0

    while not stack.isEmpty():
        state, actions, cost = stack.pop() # Pop the next state, actions, and cost from the stack
        if problem.isGoalState(state):
            return actions # Return the actions if the current state is a goal state
        if state not in visited:
            visited.add(state) # Add the current state to the visited set
            successors = problem.getSuccessors(state)
            for nextState, nextAction, nextCost in successors:
                nextActions = actions + [nextAction] # Add the next action to the action list
                nextCosts = cost + nextCost # Update the cost
                stack.push((nextState, nextActions, nextCosts)) # Add the next state, action list, and cost to the stack

    return [] # Return an empty list if no path is found
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    
    visited = set()
    queue = util.Queue()
    queue.push((problem.getStartState(), [], 0))

    while not queue.isEmpty():
        state, actions, cost = queue.pop()
        if problem.isGoalState(state):
            return actions
        if state not in visited:
            visited.add(state) # Add the current state to the visited set
            successors = problem.getSuccessors(state)
            for nextState, nextAction, nextCost in successors:
                nextActions = actions + [nextAction] # Add the next action to the action list
                nextCosts = cost + nextCost # Update the cost
                queue.push((nextState, nextActions, nextCosts)) # Add the next state, action list, and cost to the queue
    return [] # Return an empty list if no path is found
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    # Store the start state of the problem
    start = problem.getStartState()
    # Create a set to keep track of visited states
    visited = set()
    # Create a priority queue to keep track of nodes to be explored
    fringe = util.PriorityQueue()
    
    # Add the start state to the priority queue with a path of no actions and a cost of 0
    fringe.push((start, [], 0), 0)
    
    while not fringe.isEmpty():
        # Pop the node with the lowest cost
        curr_node, path, curr_cost = fringe.pop()
        
        # Skip the current node if it has already been visited
        if curr_node in visited:
            continue
        
        visited.add(curr_node)
        
        # Return the current path if the current node is a goal state
        if problem.isGoalState(curr_node):
            return path
        
        # Expand the current node and add its successors to the queue
        for next_node, action, cost in problem.getSuccessors(curr_node):
            if next_node not in visited:
                # Calculate the new path and cost
                next_path = path + [action]
                next_cost = curr_cost + cost
                # Add the next node to the queue with its path and cost
                fringe.push((next_node, next_path, next_cost), next_cost)
    # Return an empty path if no solution is found
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    # Store the start state of the problem
    start_state = problem.getStartState()
    # Create a priority queue to keep track of nodes to be explored
    frontier = util.PriorityQueue()
    # Add the start state to the priority queue with a path of no actions and a cost of 0
    frontier.push((start_state, [], 0), 0)

    # Create a set to keep track of explored nodes
    explored = set()

    while not frontier.isEmpty():
        # Pop the node with the lowest combined cost and heuristic
        state, actions, cost_so_far = frontier.pop()

        # Skip the current node if it has already been explored
        if state in explored:
            continue
        
        # Return the current path if the current node is a goal state
        if problem.isGoalState(state):
            return actions

        explored.add(state)

        # Expand the current node and add its successors to the queue
        for next_state, action, step_cost in problem.getSuccessors(state):
            # Calculate the new cost and heuristic value
            new_cost = cost_so_far + step_cost
            heuristic_cost = heuristic(next_state, problem)
            priority = new_cost + heuristic_cost
            # Add the next node to the queue with its path, cost, and heuristic value
            frontier.push((next_state, actions + [action], new_cost), priority)
    # Return None if no solution is found
    return None
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
