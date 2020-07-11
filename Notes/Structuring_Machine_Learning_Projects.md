# Structuring Machine Learning Project
## Week 1:
<ul>Ways to improve our Machine learning algorithm 
	<li>Collect more data</li>
	<li>COllect more diverse data</li>
	<li>Train algorithm longer using gradient descent</li>
	<li>Try different optimisation method like Adam</li>
	<li>Try bigger network</li>
	<li>Try smaller network</li>
	<li>Try dropout</li>
	<li>L2 regularisation</li>
	<li>Changing network architecture like activation functions, number of layers etc</li>
</ul>

### Orthogonalisation in ML:
* Not fit training set well on the cost function
* Not fit _dev set well on the cost function
* Not fit test set well on the cost function
* Not generalising to real world examples

### F1 Score:
<p>Precision is defined as:</p>

``` python
p = true_positives/(predicted_positives)
r = true_positives/(positives)
F1_score = 2 *p*r/(p+r)
```
### Optimizing and Satisficing Metrics:
* Satisficing Metric - We do not worry about the value of these metric as long as it is inside acceptable bounded values.
* Optimizing metric - We want to have the best value of these as long as the satisficing metric satisfy their conditions.

### Choosing Dev and Test Set:
* Choose dev set and test set to reflect data you expect to get in the future and consider important to do wel on.
* Use about max 10000 for dev and test set each (in case of large data, ie >1000000)
* For lesser amount of data, 60/20/20 is ideal.
* If we do not need to know how the algorithm would generalise to newer data, we may skip the test set.
* When to change our evaluation metric/dev set/test set:
	1. If doing well on your metric + dev/test set does not correspond to doing well on real world data.

### Comparing to Human Level Performance:
<p>Least error possible theoretically = Bayes optimal error
This is caused due to inaccuracies in the training set, which no human or machine can predict. Reasons for reduce in the rate of increase in the accuracies of the algorithm, after it crosses the human level performance:</p> 
* Human level performance is very close to Bayes optimal error.
* When algorithm performs worse than human level performance, we can still feed it human labelled data to improve it/manual error analysis/ better analysis of bias/variance.
* Rate of improvement of accuracies will increase at a faster rate when it is yet to surpass the human level performance.
* When it is still below human level of performance we can also manually analyse error.

### Avoidable Bias:
<p>Human level error can be considered as approximately Bayes error.
<br>
Avoidable bias is the difference between the train set error and the human level error.
<br>	
Comparing the training and human level error, we can conclude whether we need to reduce avoidable bias or the variance depending on whether significant avoidable bias is present or not.</p>

### Improving performance for supervised learning:
<p>Assumptions:</p>
* We can reduse avoidable bias.
* Training set generalises well to dev set and test set

<p>To reduce avoidable bias:</p>
* Train bigger model
* Train longer/better optimisation algorithm (like RMSprop/momentum/Adam)
* Change NN architecture
<br>
<p>To reduce Variance:</p>
* More data
* Regularisation
* Change NN architecture

## Week 2:
### Error Analysis:
#### Mislabelled Examples
* Mislabelled data means data which was predicted wrongly by the algorithm
<p>When we have errors, it is advisable to go through the mislabelled examples in the dev set. We may find many different types of images that are being mislabelled. But we must ask the question "Is it worth  to spend a lot of time on that particular type of error?". To answer this question we can look at some 100 misclassified images and classify them in categories in which they belong. Then look at which category is causing the max percentage of error and use our time on improving on that category of data. For eg, in a cat classifier, it may misclassify some dogs as cats. If such images causes around 50% of the error then it is advisable to work on the classifying the dogs as non cats. If the dogs are just 5% then it is not worth the time.</p>

#### Incorrectly Labelled Data:
* Incorrectly data means data which was wrongly labelled in the data itself.
<p>Again here to see whether it is worth the time to go on correcting labels on the dev/test set, follow the psame process that we had to do for mislabelled data.
<br>
For mislabelled data in the training set, as long as the mislabelled data is just due to random errors and it is not systematically made, it would not do much harm to the algorithm.</p>
* Apply the same process to the dev and test set to make sure they come from the same set.
* Consider examining examples your algorithm got right as well as ones it got wrong.
* Train and dev/test data may come from slightly different distributions.

### Tips to build a new ML system:
* Set up dev/test set and metric
* Build initial system quickly
* Then use bias/variance  and error analysis to prioritize next step.

### Mismatched Training and Dev/Test set:
* The dev and test set should have only data which we expected to get.
* If we have some amount of data which we expect to get and much more of data which we found online/purchased data, the training set will comprise of all the purchased/online data and some fraction of the data which we expect to get, and the rest of the data can be split into dev and test set.
* Split the training set as training set and the training-dev set. These two are from the same set ie the original training set is split randomly between training seta nd the training-dev set. 
* Now training is done only on the training set and the algorithm doesn't see te training-dev set.
* Now avoidable bias is defined as the difference between human level performance and the training set.
* Variance is defined as the difference between error on the training set and the training-dev set.
* Data mismatch error is defined as the difference between the error on training-dev set and the dev set error.

<table>
	<th> </th>
	<th>General Speech recognition</th>
	<th>Rear-view speech data</th>
	<tr>
		<td><b>Human Level</b></td>
		<td>"Human level" 4%</td>
		<td> - </td>
	</tr>
	<tr>
		<td><b>Error on examples trained on</b></td>
		<td>"Training Error" 7%</td>
		<td> - </td>
	</tr>
	<tr>
		<td><b>Error on examples <u>NOT</u> trained on</b></td>
		<td>"Training-dev" 10%</td>
		<td> "Dev/Test error" 6% </td>
	</tr>
</table>

#### Addressing Data Mismatch:
* Manual error analysis to try to understand the differences between the training and the dev sets.
* Make training data more similar or collect more data similar to the dev/test sets.
* While synthesizing artificial beware that we must not have our examples from just one subset of the total possible set. If we just have a subset, the algorithm might overfit to that subset.

### Transfer Learning:
* Transferring a pretrained network for doing a particular task A, and after that we train it on data set for task B, by just modifying the later part of the network.
* Transfer learning is used when :
	* Task A and Task B have same input 
	* Transfer learning is used when we have much more data for task A and relatively less data for task B.
	* Each example of task B is much more valuable than each example of task B.
	* Low level features learnt during training of task A is useful for Task B

### Multi-Task Learning:
* To classify one input to many outputs. The output vector need not be a one hot vector, ie it can belong to multple classes as well.
* The data may not be labelled for all classes. It should be labelled(+ve or -ve) for atleast one class.
* We use logistic cost function and we sum over the classes(which have been labelled for that particular class).
* When to use multitask transfer learning:
	* Training  on a set of tasks that could benefit from having shared lower level features.
	* (not necessarily) Amount of data for each task is quite similar.
	* We can train a big enough neural network to do well on all the tasks.

### End to End Deep Learning:
* Input to output, everything happens in the same NN, unlike traditional pipelie approach where we had to train it detect features, then use these to detect more complex features, and so on.
* Usually we use end to end DL when we have large enough data.
* Traditional methods require lesser amount of data.
* Pros :
	* We need not determine what all features the network needs to learn to complete the task and we can let the NN decide it for itself.
	* Less hand designed components.
* Cons:
	* Needs a large amount of data.
	* This excludes potentially useful hand designed components.
* When to use:
	* When we have sufficient data to learn a function of the complexity needed to map X to Y.