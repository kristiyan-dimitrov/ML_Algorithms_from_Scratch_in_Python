## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

## Free response questions (5 points)

1. (0.5 points) Assume you have a K-nearest neighbor (KNN) classifier for doing a 2-way classification. Assume it uses an <img src="/tex/09af92d48ab87fa468ebde78082d1091.svg?invert_in_darkmode&sanitize=true" align=middle width=17.96371994999999pt height=22.465723500000017pt/> norm distance metric (e.g. Euclidean, Manhattan, etc.). Assume a naive implementation, like the one taught to you in class. What is the time complexity to select a class label for a new point using this model? Give your answer as a function of the number of points, <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, in the data set. What is the space complexity of the model, in terms of <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> and the number of dimensions <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/> in the vector representing each data point? Explain your reasoning. 

2. (0.25 points) What is the time-complexity of training a KNN classifier, in terms of the number of points in the training data, <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, and the number of dimensions <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/> in the vector representing each data point? Explain your reasoning.

3. (0.25 points) Is a KNN classifier able to learn a non-linear decision surface without using a data transformation prior to inputting the data to the classifier? Why or why not? 

4. (0.25 points) Assume a Euclidean distance metric. Give an illustration of where the function produced by KNN regression would be very similar to that of a learned linear regressor, but very different when considered on a larger range of input values than exists in the training data. Provide a graph and an explanation. 

5. (0.25 points) We will build up a *collaborative filter* over the next few questions. The questions presume that you have implemented the `collaborative_filtering` function and passed the corresponding test case `test_collaborative_filtering`. Collaborative filters are essentially how recommendation algorithms work on sites like Amazon ("people who bought blank also bought blank") and Netflix ("you watched blank so you might also like blank"). They work by comparing distances between users. If two users are similar, then items that one user has seen and liked but the other hasn't seen are recommended to the other user. First, frame this problem as a KNN regression problem. How are users compared to each other? Given the K nearest neighbors of a user, how can these K neighbors be used to estimate the rating for a movie that the user has not seen?

6. (0.25 points) The MovieLens dataset contains 100k ratings on 1682 movies given by 942 users. First, explore the MovieLens data. For each pair of users, what is the mean number of movies that two users have reviewed in common? What is the median number of movies that two reviewers have reviewed in common? How would the number of movies that two users have both reviewed affect the quality of your collaborative filter? What occurs for those movies that no user has rated? 

#### The following questions presume that you have implemented the `collaborative_filtering` function and passed the corresponding test case `test_collaborative_filtering`. You will now need to create additional code to let you explore the effect of different design choices on the performance of a collaborative filter. We will not evaluate this code. We will evaluate the experimental results you produce using this code. The process for this is as follows (note the free parameters `N, K, D, A`):
   * Once you load the movielens dataset using your `load_movielens_data` function, replace all 0 values for each user with median value of the user's ratings (when computing median, you should only consider non-zero values).
   * For each user in MovieLens, take `N` *real* ratings at random (the median values you newly filled in the previous step is *NOT real* rating) and replace them with 0. Save this altered data as a new matrix. Be sure to keep the original data around to evaluate your approach.
   * Use the `collaborative_filtering` function to impute values for this new matrix with `n_neighbors = K`, `distance_measure = D`, and `aggregator = A`. 
   * To evaluate the quality of the collaborative filter, use rooted mean squared error For each *known* rating that you estimated, do: (original_rating - estimated_rating) ** 2. Take the mean of all of these, and then take the square root of that. This is the rooted mean squared error evaluation: error = sqrt(mean((r_i - rhat_i)^2))  where r_i is the original rating (before you substitued a 0) and rhat_i is the rating your system produced, and i iterates over all of the ratings you replaced with 0. Do not estimate the error on ratings that were 0 before you altered them.

   HINT: Write ONE function that takes in `(N, K, D, A)` as arguments and returns the error of a collaborative filter on the MovieLens dataset. You will be able to use this function for each of the following questions.

   WARNING: Some of these plots can take a very long time to generate as the collobarative filter process is *expensive*. Start early so your computations finish before the deadline! The runtime of the solution (run on a Macbook Pro 2018 laptop) is noted at the end of each question. One tip is to first run the experiments on a small subset of the data to make sure everything works, say just the first 100 users. Once everything works, move up to the entire dataset of 943 users.
 
7.  (0.5 points) For this question, we will measure the effect of changing `N`, the number of ratings we blank out per user. Report the error of your collaborative filter for the values `N` = 5, 10, 20. Keep `K`, `D`, and `A` fixed at 3, `'euclidean'`, and `'mean'`, respectively. Report the error via a **plot**. On the x-axis should be the value of N and on the y-axis should be the rooted mean squared error of imputating missing values. What happens as N increases? (Estimated runtime: 95 seconds)

8. (0.5 points) For this question, we will measure the effect of changing `D`, the distance measure we use to compare users. Report the error of your collaborative filter via a **table**. For each possible distance measure, report the error of the filter. Keep `N`, `K`, and `A` fixed at 1, 3, and `'mean'`, respectively. Report the error for each possible distance measure `'euclidean'`, `'cosine'`, and `'manhattan'` distance. (Estimated runtime: 52 seconds)
    
9. (0.5 points) Once you find the best performing distance measure, measure the effect of `K`. Report the error of your collaborative filter via a **plot**. On the x-axis should be the value of `K` and on the y-axis should be the rooted mean squared error of the filter. Do this for `K` = 1, 3, 7, 11, 15, 31. Keep `N`, `D`, and `A` fixed at 1, the best performing distance measure found in the previous question, and `'mean'`, respectively. (Estimated runtime: 95 seconds)
    
10. (0.5 points) Finally, measure the effect of `A`, the aggregator used. Report the error of your collaborative filter via a **table**. For each possible aggregator, report the error of the filter. Keep `N`, `D`, and `K` fixed at 1, the best performing distance measure you found, and the best performing value of `K` that you found, respectively. Set A to be either `'mean'`, `'mode'`, or `'median'`. (Estimated runtime: 110 seconds). 
  
11. (0.25 points) Now, discuss your results. What were the best settings you found for `D`, `A`, and `K`? For `D`, why do you think that distance measure worked best? For `A`, what about the best setting made it more suitable for your dataset? How did the value of `K` affect your results? Engage critically with the results of your experiments. For each graph you generated, propose an explanation for the behavior of that graph. For each table of values, propose an explanation for the variation in your results.

 12. (0.25 points) Your boss gives you a very large dataset. What problems do you see coming up with implementing kNN? Name a few of them.
 
 13. (0.25 points) Discuss possible solutions to the problems you have identified in question (12).
 
 14. (0.25 points) Given six training examples as ((a_i, b_i), y_i) values, where a_i and b_i are the two feature values (positive integers) and y_i is the class label: {((1,1),-1),((2, 2), −1), ((2, 8), +1), ((4, 4), +1), ((6, 5), −1), ((3, 6), −1)}. Classify the following test example at coordinates (4, 7) using a k-NNclassifier with k = 3 and Manhattan-distance defined by d((u, v), (p, q)) = |u −p| + |v −q|. Explain shortly how you came up with the answer.
 
 15. (0.25 poins) Assume we increase k in a k-NN classifier from 1 to n, where n is the number of training examples. Discuss if the classification accuracy on the training set increases. Consider weighted and unweighted K-NN classifiers.

