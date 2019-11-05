There are many cycles seperated or not。Their centers are on a line. Can you calculate how many seperated relations among these cycles?     
    
input:    
    int[] centers    
    int[] radiuses    
output:    
    int n    

example:    
    centers=[1,2,3,4,5,7,8]    
    radiuses=[1,1,2,3,2,2,2]    
output:5    
    seperated relations [1,5],[1,7],[1,8],[2,7],[2,8]    
    start_end[[0,2],[1,3],[1,5],[1,7],[3,7],[5,9],[6,10]]    

![avatar](cycles.jpg)    