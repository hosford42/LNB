Less Naive Bayes
================
Copyright (c) Aaron Hosford 2013-2015, all rights reserved.

Less Naive Bayes is a novel classification algorithm introduced in this
module, which extends the standard Naive Bayes classification algorithm to
improve classification accuracy. This module implements both Naive Bayes
(NB) and Less Naive Bayes (LNB) classifiers. In addition, it implements
another novel variant of Naive Bayes called Inertial Naive Bayes (INB),
also introduced in this module, which is similar to NB except that it is
intentionally biased to repeat the same misclassifications when it is
incapable of correctly classifying all inputs, rather than oscillating
as standard NB does.

LNB works by successively training new INB classifiers to predict both
the target classification and the misclassifications of earlier
classifiers. By doing this, it successively subdivides the feature space
into linearly separable subspaces which INB (and NB) are better capable
of distinguishing. Eventually the feature space is transformed
sufficiently that the most recent layer is fully capable of
correctly classifying all samples with distinguishable features.

As an example, let's look at the ordinary XOR function. Suppose we have
two features which the classifier is permitted to observe, A and B, and
these features may either be present or absent. The classifier is
expected to predict whether a given sample is accepted or rejected. Let's
suppose that a sample is to be accepted if both A and B are present, or
both A and B are absent, and that a sample is to be rejected if either
A is present and B is not, or B is present and A is not. For clarity,
here is a table laying out the appropriate classifications for each
sample:

      A   B   | Accept
      --------+-------
      No  No  | Yes
      Yes No  | No
      No  Yes | No
      Yes Yes | Yes

Ordinary Naive Bayes is not capable of learning this classification
problem with 100% accuracy. At least one of the four types of samples
will be misclassified no matter how many training samples are provided,
and no matter what the proportions are of those samples in the sample
space. The reason is that for Naive Bayes to separate two types of
samples into distinct categories, the boundary for separating the
categories must coincide with the boundary of at least one feature. As
you can see, looking at the presence of A alone does not provide
sufficient information to isolate any subset of the samples that belongs
to a particular category from the others. Likewise for B. Only by looking
at the combined values of A and B can we isolate a subset of the samples
which should all be accepted or rejected. Since Naive Bayes assumes the
features are independent of each other with respect to the classification
categories, this means that Naive Bayes cannot look at specific
combinations of values, and is therefore incapable of distinguishing the
categories. However, Naive Bayes is capable of correctly classifying 3
out of the 4 sample types, while misclassifying the 4th.

The standard Naive Bayes algorithm will, depending on the frequencies of
the samples, continue to shift which sample type in the above table is
the one that is incorrectly classified. But by making a minor adjustment
to the standard Naive Bayes algorithm, we can cause it to stabilize and
consistently misclassify the same type of sample every time. This is done
by adjusting the weight of observations that agree with the current
classification to be greater than the weight of observations that
disagree. This is the Inertial Naive Bayes algorithm.

Now, what is the point in making the errors consistent for a particular
sample type? Let's make a new table, this time with both the original
target classifications, and with the classifications made by an INB
classifier which has settled on misclassifying samples that match the
top row:

      A   B   | Accept  INB
      --------+------------
      No  No  | Yes     No
      Yes No  | No      No
      No  Yes | No      No
      Yes Yes | Yes     Yes

Observe that the INB classifier can indeed learn the classifications
assigned to it in this table, by weighting No's more heavily for both
A and B. Now let's take the unique combinations of the two right-hand
columns as classifications in their own right. Here is a table assigning
a unique number to each unique combination of categories:

      A   B   | Accept  INB  | Combined
      --------+--------------+-----------
      No  No  | Yes     No   | 1 (Yes/No)
      Yes No  | No      No   | 2 (No/No)
      No  Yes | No      No   | 2 (No/No)
      Yes Yes | Yes     Yes  | 3 (Yes/Yes)

There are three unique combinations of target and INB classifications,
each with its own assigned number. 2 appears twice because on both rows,
we have the same combination of No/No, while 1 and 3 correspond to
combinations that appear only once each. Now if we look at boundaries
between the categories on the far right and compare them to boundaries
between the features on the far left, we can see that the features can
indeed be used to distinguish each of the three combined categories.

Category 1 can be distinguished from category 2 by more heavily weighting
Yes's for each feature in favor of 2. Category 3 can be distinguished
from category 2 by more heavily weighting No's for each feature in favor
of 2. Thus the classifications for the new, derived problem are linearly
separable, which means that a Naive Bayes (or Inertial Naive Bayes)
classifier can learn this new problem with 100% accuracy.

Now, once we have a classifier that has learned the new, derived problem,
we can construct a classifier that correctly solves the original
classification problem by mapping from categories 1 and 3 to Yes and
from category 2 to No. By consecutively training two INB classifiers and
using the outputs from the first classifier to augment the target
classifications before they are presented to the second classifier, we
transform the problem into one the second classifier can learn. Then we
construct the correct classifier from the second INB classifier. For more
difficult problems, we can repeat this iterative augmentation, using the
classifications produced by all earlier classifiers to augment the target
classifications for each successive classifier. This, together with a
stopping condition that detects when further iterations will no longer
produce improved performance, constitutes the Less Naive Bayes algorithm.
