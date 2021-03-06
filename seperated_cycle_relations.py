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
[1,5],[1,7],[1,8],[2,7],[2,8]


solution by Heyi

import bisect
def query_circle(points):
   right = sorted([p[1] for p in points])
   return sum(bisect.bisect_left(right, p[0]) for p in points)
