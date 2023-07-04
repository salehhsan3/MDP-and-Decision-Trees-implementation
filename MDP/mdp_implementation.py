from copy import deepcopy
import numpy as np
from mdp import MDP # included by saleh, maybe remove before submission
from copy import deepcopy

def actionUtility(mdp: MDP, U, curr_state, action):
    potential_util = [ U[new_row][new_col] for (new_row, new_col) in [ mdp.step( (curr_state[0],curr_state[1]), action ) for action in list(mdp.actions.keys()) ] ]
    probabilities = list( mdp.transition_function[action] )
    return sum( [ p * u for p, u in zip(probabilities, potential_util) ] )

def value_iteration(mdp: MDP, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    delta = np.inf # to initially enter the loop
    _Utilities = [[0 for x in range(mdp.num_col)] for y in range(mdp.num_row)]
    Utilities = None
    for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
            if (state_row, state_col) in mdp.terminal_states:
                _Utilities[state_row][state_col] = float( mdp.board[state_row][state_col] ) # may affect calculations in next loop
            elif mdp.board[state_row][state_col] == 'WALL':
                _Utilities[state_row][state_col] = None # may affect calculations in next loop
    
    while delta > ( ( epsilon * ( 1 - mdp.gamma ) ) / mdp.gamma ):
        delta = 0
        Utilities = deepcopy(_Utilities)
        for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
            if (state_row, state_col) in mdp.terminal_states:
                continue
            elif mdp.board[state_row][state_col] == 'WALL':
                continue
            else:
                val_list = [ actionUtility(mdp, Utilities,(state_row, state_col), action) for action in list(mdp.actions.keys()) ]
                _Utilities[state_row][state_col] = float( mdp.board[state_row][state_col] ) + ( mdp.gamma * max( val_list ) )
                delta = max( delta, abs( _Utilities[state_row][state_col] - Utilities[state_row][state_col]))
    return Utilities
    # ========================


def get_policy(mdp: MDP, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(U)
    for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
            if (state_row, state_col) in mdp.terminal_states:
                policy[state_row][state_col] = None
            elif mdp.board[state_row][state_col] == 'WALL': 
                continue 
            else:
                expectancy = [ actionUtility(mdp, U, (state_row, state_col), action) for action in list(mdp.actions.keys()) ] 
                act_ind = expectancy.index(max(expectancy))
                policy[state_row][state_col] = list(mdp.actions.keys())[act_ind]
    return policy
    # ========================


def policy_evaluation(mdp: MDP, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    _Utilities = [[0 for x in range(mdp.num_col)] for y in range(mdp.num_row)]
    Utilities = None
    epsilon = 10 ** (-3)
    delta = np.inf
    for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
        if (state_row, state_col) in mdp.terminal_states:
            _Utilities[state_row][state_col] = float( mdp.board[state_row][state_col] )
        elif mdp.board[state_row][state_col] == 'WALL':
            _Utilities[state_row][state_col] = None
            
    while delta > ( epsilon * (1-mdp.gamma) / mdp.gamma ):
        Utilities = deepcopy(_Utilities)
        delta = 0
        for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
            if (state_row, state_col) in mdp.terminal_states:
                continue
            elif mdp.board[state_row][state_col] == 'WALL':
                continue
            else:
                _Utilities[state_row][state_col] = float( mdp.board[state_row][state_col] ) + ( mdp.gamma *  actionUtility(mdp, Utilities, (state_row, state_col), policy[state_row][state_col] ) )
                delta = max( delta, abs(_Utilities[state_row][state_col] - Utilities[state_row][state_col]) )
    return Utilities
    # ========================


def policy_iteration(mdp: MDP, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    unchanged_policy = False
    policy = deepcopy(policy_init)
    while unchanged_policy == False:
        Utilities = policy_evaluation(mdp, policy)
        unchanged_policy = True
        for state_row, state_col in [ (x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]:
            if (state_row, state_col) in mdp.terminal_states:
                continue
            elif mdp.board[state_row][state_col] == 'WALL':
                continue
            else:
                val_list = [ actionUtility(mdp, Utilities,(state_row, state_col), action) for action in list(mdp.actions.keys()) ]
                if max( val_list ) > actionUtility(mdp, Utilities,(state_row, state_col), policy[state_row][state_col]):
                    policy[state_row][state_col] = list(mdp.actions.keys())[val_list.index( max(val_list) )]
                    unchanged_policy = False
    return policy
    # ========================
