#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A,
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like:
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List







'''
 pseudo code as below:

                 ransacMatching
 Input  : two list of list donating 2 sets of points
 Output : the homograph matrix between 2 sets

 -------------------------------------------------
 k := max_iter
 best_matrix
 cnt_inliers_max := 0
 while k >0 do
    sample = getRandomPointFromAB(A,B)
    homo_matrix = getHomograph(sample)
    inliers = sample
    transformed_a_by_homograph = transform(homo_matrix,A)
    for point in transformed_a_by_homograph:
        err = distance(point,point_B)
        if err < dis_threshold :
            inliers.append((point,pointB)
    if len(inliers) > cnt_inliers_max :
        cnt_inliers_max = len(inliers)
        best_matrix = homo_matrix
    k--
  return best_matrix
'''
