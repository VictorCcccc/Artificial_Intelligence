from time import sleep
import math
from math import inf
from random import randint
import copy
import sys

class ultimateTicTacToe(object):

    def pattern(self, boardIdx):
        c = self.globalIdx[boardIdx]
        return [
            [c, (c[0]+1, c[1]), (c[0]+2, c[1])], #row1
            [(c[0], c[1]+1), (c[0]+1, c[1]+1), (c[0]+2, c[1]+1)], #row2
            [(c[0], c[1]+2), (c[0]+1, c[1]+2), (c[0]+2, c[1]+2)], #row3
            [c, (c[0], c[1]+1), (c[0], c[1]+2)], #col1
            [(c[0]+1, c[1]), (c[0]+1, c[1]+1), (c[0]+1, c[1]+2)], #col2
            [(c[0]+2, c[1]), (c[0]+2, c[1]+1), (c[0]+2, c[1]+2)], #col3
            [c, (c[0]+1, c[1]+1), (c[0]+2, c[1]+2)], #x1
            [(c[0]+2, c[1]), (c[0]+1, c[1]+1), (c[0], c[1]+2)], #x2
        ]

    def __init__(self):
        """
        Initialization of the game.
        """
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        self.maxPlayer = 'X'
        self.minPlayer = 'O'
        self.maxDepth = 3
        #The start indexes of each local board
        self.globalIdx = [(0, 0), (0, 3), (0, 6), (3, 0),
                          (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx = 4
#        self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility = 10000
        self.twoInARowMaxUtility = 500
        self.preventThreeInARowMaxUtility = 100
        self.cornerMaxUtility = 30

        self.winnerMinUtility = -10000
        self.twoInARowMinUtility = -100
        self.preventThreeInARowMinUtility = -500
        self.cornerMinUtility = -30

        self.expandedNodes=0
        self.currPlayer=True


    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')


    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        utility, applied = self.firstRule(isMax)
        if applied:
            return utility
        utility, applied = self.secondRule(isMax)
        if applied:
            return utility
        return self.thirdRule(isMax)


    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        utility, applied = self.firstRule(isMax)
        if applied:
            return utility
        utility, applied = self.secondRule(isMax)
        if applied:
            return utility
        return self.selfRule()

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        for x in range(9):
            for y in range(9):
                if self.board[x][y] == '_':
                    return True
        return False

    def getElem(self, pos):
        return self.board[pos[0]][pos[1]]

    def countPlayerTokens(self, pat, token):
        count = 0
        for pos in pat:
            if self.getElem(pos) == token:
                count += 1
        return count

    def checkWinner(self):
        #Return terminal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        for num in range(9):
            for pat in self.pattern(num):
                if self.getElem(pat[0]) == self.getElem(pat[1]) == self.getElem(pat[2]):
                    if self.getElem(pat[0]) == self.maxPlayer:
                        return 1
                    if self.getElem(pat[0]) == self.minPlayer:
                        return -1
        return 0

    def firstRule(self, isOffensive):
        utility = 0
        winner = self.checkWinner()
        if winner == 1:
            utility = 10000 if isOffensive else 0
        elif winner == -1:
            utility = -10000 if not isOffensive else 0
        else:
            utility = 0
        return utility, (utility != 0)

    def secondRule(self, isOffensive):
        utility = 0
        applied = False
        myPlayer = self.maxPlayer if isOffensive else self.minPlayer
        opponentPlayer = self.minPlayer if isOffensive else self.maxPlayer
        for num in range(9):
            for pat in self.pattern(num):
                if self.countPlayerTokens(pat, myPlayer) == 2 \
                    and self.countPlayerTokens(pat, '_') == 1:
                    utility += 500 if isOffensive else -100
                    applied = True
            for pat in self.pattern(num):
                if self.countPlayerTokens(pat, opponentPlayer) == 2 \
                    and self.countPlayerTokens(pat, myPlayer) == 1:
                    utility += 100 if isOffensive else -500
                    applied = True
        return utility, applied

    def thirdRule(self, isOffensive):
        utility = 0
        for num in range(9):
            top_left = self.globalIdx[num]
            top_right = (top_left[0], top_left[1]+2)
            bottom_left = (top_left[0]+2, top_left[0])
            bottom_right = (top_left[0]+2, top_left[1]+2)
            corners = [top_left, top_right, bottom_left, bottom_right]
            corner_num = self.countPlayerTokens(corners, self.maxPlayer if isOffensive else self.minPlayer)
            utility += corner_num * (30 if isOffensive else -30)
        return utility

    def selfRule(self):
        utility = 0
        for num in range(9):
            top_left = self.globalIdx[num]
            top_right = (top_left[0], top_left[1] + 2)
            bottom_left = (top_left[0] + 2, top_left[0])
            bottom_right = (top_left[0] + 2, top_left[1] + 2)
            corners = [top_left, top_right, bottom_left, bottom_right]
            center = [(top_left[0] + 1,top_left[1] + 1)]
            corner_num_self= self.countPlayerTokens(corners, self.minPlayer)
            corner_num_opp = self.countPlayerTokens(corners, self.maxPlayer)
            center_num_opp = self.countPlayerTokens(center,self.maxPlayer)
            center_num_self = self.countPlayerTokens(center,self.minPlayer)
            utility += (corner_num_self * (-30) + center_num_self * (-70) + center_num_opp * (50) + corner_num_opp *(10))
        return utility

    def alphabetePre(self, depth, currBoardIdx, alpha, beta, isMax):

        if self.checkMovesLeft() == False:
            return self.evaluateDesigned(isMax)

            # if depth == 3, return utility.
        if depth == 3:
            return self.evaluateDesigned(isMax)
        # get list of legal moves for self.board

        legalMoves = self.getLegalMoves(currBoardIdx, isMax)

        if len(legalMoves) == 0:
                # tie
            return 0

        # for each move
        for move, newBoardIdx in legalMoves:
            #  update self.board with that move
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            # call minimax with the appropriate parameters
            v = self.alphabeta(depth + 1, newBoardIdx, alpha, beta, not isMax)
            # update bestValue
            if isMax:
                bestValue = max(bestValue, v)
                alpha = max(alpha, bestValue)
            else:
                bestValue = min(bestValue, v)
                beta = min(beta, bestValue)

            # undo the modification to self.board
            self.board[move[0]][move[1]] = '_'
            if alpha >= beta:
                return bestValue

        # return bestValue
        return bestValue

    def alphabeta(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        bestValue = -math.inf if isMax else math.inf
        self.expandedNodes += 1
        # if no moves left, return utility
        if self.checkMovesLeft() == False:
            return self.evaluatePredifined(isMax)

        # if depth == 3, return utility.
        if depth == 3:
            return self.evaluatePredifined(isMax)
        # get list of legal moves for self.board
        legalMoves = self.getLegalMoves(currBoardIdx, isMax)

        if len(legalMoves) == 0:
            # tie
            return 0

        # for each move
        for move, newBoardIdx in legalMoves:
            #  update self.board with that move
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            # call minimax with the appropriate parameters
            v = self.alphabeta(depth + 1, newBoardIdx, alpha, beta, not isMax)
            # update bestValue
            if isMax:
                bestValue = max(bestValue, v)
                alpha = max(alpha, bestValue)
            else:
                bestValue = min(bestValue, v)
                beta = min(beta, bestValue)
            # undo the modification to self.board
            self.board[move[0]][move[1]] = '_'
            if alpha >= beta:
                return bestValue

        # return bestValue
        return bestValue
    def alphabetaPre(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        bestValue = -math.inf if isMax else math.inf
        self.expandedNodes += 1
        # if no moves left, return utility
        if self.checkMovesLeft() == False:
            return self.evaluateDesigned(isMax)

        # if depth == 3, return utility.
        if depth == 3:
            return self.evaluateDesigned(isMax)
        # get list of legal moves for self.board
        legalMoves = self.getLegalMoves(currBoardIdx, isMax)

        if len(legalMoves) == 0:
            # tie
            return 0

        # for each move
        for move, newBoardIdx in legalMoves:
            #  update self.board with that move
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            # call minimax with the appropriate parameters
            v = self.alphabeta(depth + 1, newBoardIdx, alpha, beta, not isMax)
            # update bestValue
            if isMax:
                bestValue = max(bestValue, v)
                alpha = max(alpha, bestValue)
            else:
                bestValue = min(bestValue, v)
                beta = min(beta, bestValue)
            # undo the modification to self.board
            self.board[move[0]][move[1]] = '_'
            if alpha >= beta:
                return bestValue

        # return bestValue
        return bestValue

    def alphabeta_wrapper(self, depth, currBoardIdx, isMax, isPre):
        if isPre:
            return self.alphabeta(1, currBoardIdx, -math.inf, math.inf, not isMax)
        else:
            return self.alphabetaPre(depth, currBoardIdx, -math.inf, math.inf, isMax)

    def getLegalMoves(self, currBoardIdx, isMax):
        legalMoves = []
        currPos = self.globalIdx[currBoardIdx]
        for x in range(3):
            for y in range(3):
                pos = (currPos[0]+x, currPos[1]+y)
                if self.getElem(pos) == '_':
                    boardIdx = 3 * x + y
                    legalMoves.append((pos, boardIdx))
        return legalMoves

    def minimax_wrapper(self, depth, currBoardIdx, isMax, isPre):
        return self.minimax(depth, currBoardIdx, isMax)


    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        bestValue = -math.inf if isMax else math.inf
        self.expandedNodes += 1
        # if no moves left, return utility
        if self.checkMovesLeft() == False:
            return self.evaluatePredifined(not isMax)

        # if depth == 3, return utility.
        if depth == 3:
            return self.evaluatePredifined(not isMax)

        # get list of legal moves for self.board
        legalMoves = self.getLegalMoves(currBoardIdx, isMax)
        if len(legalMoves) == 0:
            # tie
            return 0

        # for each move
        for move, newBoardIdx in legalMoves:
            #  update self.board with that move
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            # call minimax with the appropriate parameters
            v = self.minimax(depth + 1, newBoardIdx, not isMax)
            # update bestValue
            if isMax:
                if v > bestValue:
                    bestValue = v
            else:
                if v < bestValue:
                    bestValue = v
            # undo the modification to self.board
            self.board[move[0]][move[1]] = '_'
        # return bestValue
        return bestValue

    def playGamePredifinedAgent(self, maxFirst, isMinimaxOffensive, isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove = []
        bestValue = []
        gameBoards = []
        expandedNodes = []
        winner = 0
        currBoardIdx = self.startBoardIdx
        isMax = maxFirst
        gameBoards.append(copy.deepcopy(self.board))
        self.printGameBoard()
        def get_algo():
            if isMax:
                return self.minimax_wrapper if isMinimaxOffensive else self.alphabeta_wrapper
            else:
                return self.minimax_wrapper if isMinimaxDefensive else self.alphabeta_wrapper
        while self.checkMovesLeft():
            winner = self.checkWinner()
            if winner != 0:
                # win or lose
                break
            legalMoves = self.getLegalMoves(currBoardIdx, isMax)
            if len(legalMoves) == 0:
                # tie
                break
            bestV1 = -math.inf
            bestV2 = math.inf
            for move, newBoardIdx in legalMoves:
                self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
                v = get_algo()(1, newBoardIdx, not isMax, isPre = True)
                self.board[move[0]][move[1]] = '_'

                if isMax:
                    if bestV1 < v:
                        bestV1 = v
                        best = (move, newBoardIdx)
                else:
                    if bestV2 > v:
                        bestV2 = v
                        best = (move, newBoardIdx)

#                if v < bestV:
#                    bestV = v

            move, newBoardIdx = best
            bestMove.append(move)
            bestValue.append(v)
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            gameBoards.append(copy.deepcopy(self.board))
            expandedNodes.append(self.expandedNodes)
            self.expandedNodes = 0
            isMax = not isMax
            currBoardIdx = newBoardIdx
            self.printGameBoard()

        return gameBoards, bestMove, expandedNodes, bestValue, winner

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        isMax = False
        bestMove = []
        bestValue = []
        gameBoards = []
        expandedNodes = []
        winner = 0
        currBoardIdx = self.startBoardIdx
        gameBoards.append(copy.deepcopy(self.board))
        self.printGameBoard()

        def get_algo():
            return self.alphabeta_wrapper

        while self.checkMovesLeft():

            winner = self.checkWinner()
            if winner != 0:
                # win or lose
                break
            legalMoves = self.getLegalMoves(currBoardIdx, isMax)
            if len(legalMoves) == 0:
                # tie
                break
            best = None
            bestV = math.inf
            for move, newBoardIdx in legalMoves:
                self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
                v = get_algo()(0, currBoardIdx, isMax, isPre = False)
                self.board[move[0]][move[1]] = '_'
                if v < bestV:
                    bestV = v
                    best = (move, newBoardIdx)
            move, newBoardIdx = best
            bestMove.append(move)
            bestValue.append(bestV)
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            gameBoards.append(copy.deepcopy(self.board))
            expandedNodes.append(self.expandedNodes)
            self.expandedNodes = 0
            isMax = not isMax
            currBoardIdx = newBoardIdx
            self.printGameBoard()

        return gameBoards, bestMove, expandedNodes, bestValue, winner


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        isMax = False
        bestMove = []
        bestValue = []
        gameBoards = []
        expandedNodes = []
        winner = 0
        currBoardIdx = self.startBoardIdx
        gameBoards.append(copy.deepcopy(self.board))
        self.printGameBoard()

        def get_algo():
            return self.alphabeta_wrapper

        while self.checkMovesLeft():
            winner = self.checkWinner()
            if winner != 0:
                # win or lose
                break
            legalMoves = self.getLegalMoves(currBoardIdx, isMax)
            if len(legalMoves) == 0:
                # tie
                break
            best = None
            bestV = math.inf
            for move, newBoardIdx in legalMoves:
                self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
                v = get_algo()(0, currBoardIdx, isMax,isPre = True)
                self.board[move[0]][move[1]] = '_'
                if v < bestV:
                    bestV = v
                    best = (move, newBoardIdx)
            move, newBoardIdx = best
            bestMove.append(move)
            bestValue.append(bestV)
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            gameBoards.append(copy.deepcopy(self.board))
            expandedNodes.append(self.expandedNodes)
            self.expandedNodes = 0
            isMax = not isMax
            currBoardIdx = newBoardIdx
            self.printGameBoard()
            BoardIdx = -1
            while currBoardIdx != BoardIdx:
                try:
                    str = input("Enter Value：")
                    row = int(str.split(",")[0])
                    col = int(str.split(",")[1])
                except:
                    print("error value, enter: x,y")
                    continue
                else:
                    move = (row, col)
                    BoardIdx = 3 * (row // 3) + col // 3
                    if BoardIdx!=currBoardIdx:
                        print("Board Error, please enter again")
            newBoardIdx = 3 * (row % 3 )+ col %3
            currBoardIdx = newBoardIdx
            self.board[move[0]][move[1]] = (self.maxPlayer if isMax else self.minPlayer)
            gameBoards.append(copy.deepcopy(self.board))
            isMax = not isMax
            self.printGameBoard()
        return gameBoards, bestMove, expandedNodes, bestValue, winner

if __name__ == "__main__":
    uttt = ultimateTicTacToe()
    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGamePredifinedAgent(
        True, True, True)
#    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGameYourAgent()
#    gameBoards, bestMove, expandedNodes, bestValue, winner = uttt.playGameHuman()
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
