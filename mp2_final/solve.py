# -*- coding: utf-8 -*-
import numpy as np

def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You can assume there will always be a solution.
    """
    shape = []
    rows = []
    pents_no = len(pents)
    board_holes = np.where(board.flatten()==0)[0]
    sol_list = []
    position = []
    for i,P in enumerate(pents):
        for P in find_orientation(P,i):
            for M in find_position(P,board,position):
                shape.append([i,P])
                c = np.zeros(pents_no)
                c[i] = 1
                rows.append(np.append(np.delete(M,board_holes),c))
    matrix = np.asarray(rows)

    index = np.arange(matrix.shape[0])
    solution = (dfs(matrix,index))
    
#    test = np.zeros(72)


    for item in solution:
        pent_shape = shape[item][1]
        pent_idx = shape[item][0]
        pent_shape[pent_shape == 1] = pent_idx+1
        row,col = position[item]
        sol_list.append([pent_shape,(row,col)])
    return sol_list

    raise NotImplementedError
'''
    for i in solution:
        test = test + matrix[i]
    print (test)
'''



def find_orientation(P,i):
    P[np.where( P > 0 )] = 1
    group = []
    for sP in (P,np.fliplr(P)):
        for i in range(0,4):
            sP = np.rot90(sP)
            s = str(sP)
            if not s in group:
                yield sP
                group.append(s)

def find_position(P,B,position):
    B_h,B_w = B.shape
    P_h,P_w = P.shape
    for i in range(B_h+1-P_h):
        for j in range(B_w+1-P_w):
            M = np.zeros((B_h,B_w),dtype='int')
            if (B[i:i+P_h,j:j+P_w].min())!=0:
                M[i:i+P_h, j:j+P_w] = P
                position.append((i,j))
                yield M.flatten()


def dfs (M,index):
    stack = []
    
    minc = np.sum(M, axis = 0).argmin()

#initial states

    for i in np.where(M[:,minc] == 1)[0]:
        del_columns = []
        del_rows = []
        new_M = M
        new_index = index 

        for k in np.where(M[i] == 1)[0]:
            del_columns.append(k)
            for j in np.where(M[:,k] == 1)[0]:
                if not j in del_rows:
                    del_rows.append(j)
        new_M = np.delete(new_M,del_columns,1)
        new_M = np.delete(new_M,del_rows,0)
        new_index = np.delete(index,del_rows)
        
        route = [index[i]]

        stack.append((route,new_index,new_M))

        

    no_solution_flag = False

    while stack:

        route,index,M = stack.pop()
        

        #print (route)

        if M.shape[1] == 0:
            if no_solution_flag == True:
                continue
            else: return route
        else:   
            if np.sum(M, axis = 0).min() == 0:
                no_solution_flag = True
                continue
            else:
                
                no_solution_flag = False

                minc = np.sum(M, axis = 0).argmin()
                for i in np.where(M[:,minc] == 1)[0]:
                    del_columns = []
                    del_rows = []
                    new_M = M
                    new_index = index 

                    for k in np.where(M[i] == 1)[0]:
                        del_columns.append(k)
                        for j in np.where(M[:,k] == 1)[0]:
                            if not j in del_rows:
                                del_rows.append(j)
                    new_M = np.delete(new_M,del_columns,1)
                    new_M = np.delete(new_M,del_rows,0)
                    new_index = np.delete(index,del_rows)
                    
                    #print (route + index[i])

                    new_route = route.copy() 

                    new_route.append(index[i])


                    stack.append((new_route,new_index,new_M))

    return None


