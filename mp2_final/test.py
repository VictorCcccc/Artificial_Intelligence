import numpy as np

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
        
        print (M)

        #print (route)

        if M.shape[0] == 0:
            if no_solution_flag == True:
                return None
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
                    route.append(index[i])

                    stack.append((route,new_index,new_M))

    return None



'''
def exact_cover(M,index):
    if M.shape[1] == 0:
        return []
    else:
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

            #print(new_M)
            partial_solution = exact_cover(new_M,new_index)

            
            
            


            print (index[i])
        return partial_solution

'''
            

            
            




        

M = np.asarray([[1,1,1,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[1,0,0,1,1,0]])


index = np.arange(M.shape[0])

M1 = np.asarray([[1,0]])




print (dfs(M,index))
