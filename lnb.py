#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
of distinguishing. Eventually these transformations transform the feature
space sufficiently that the most recent layer is fully capable of
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
"""

# TODO:
# - Use my new term, "category forking", in the documentation to describe
#   what this algorithm does.
# - It should be possible to apply category forking to other classification
#   algorithms, including neural networks. Write up and publish a paper
#   describing category forking and how to use it to implement "wide
#   learning" in ANNs -- i.e. by adding a new output node to the network,
#   adding another layer after the outputs to sum them up appropriately,
#   and then training the new node to predict when the old network would
#   produce the incorrect/correct classification. (This is a very imprecise
#   description, of course.)
# - Improve logic for addition of new layers to keep them to the bare
#   minimum required for good performance, since this drastically affects
#   the speed of training.
# - Consider revamping for off-line training: Train each layer on full data
#   set, and then determine whether to add a new layer, revert to an
#   earlier layer, or stop.
# - It appears that using an INBClassifier with bias of 0 works better.
#   Do some intensive tests to verify. Consider setting defaults
#   accordingly, and revamping documentation.
# - Implement non-categorical (linear) conditions.
# - Modify NB and INB implementations to work with log probabilities
#   instead of raw counts.
# - Allow features to be accompanied with counts to facilitate text
#   classification.
# - Restrict categories to be integers instead of arbitrary values and
#   use this to effect a speedup by representing category sequences as
#   bitstrings.
# - Wrap a classifier as a spam filter, using integer assignments
#   to tokens to speed up performance.

from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):
    """Abstract base class which defines the interface for trainable
    classifiers."""

    @abstractmethod
    def observe(self, features, category):
        """Observe a new (feature set, category) pair and update the
        classifier's model of the sample space. Features should be passed
        in as an iterable or iterator of (name, value) pairs, or of unique
        values representing (name, value) pairs."""
        raise NotImplementedError()

    @abstractmethod
    def probabilities(self, features):
        """Given a set of features, return a dictionary mapping each
        category to its respective probability as predicted by the
        classifier's model of the sample space. Features should be passed
        in as an iterable or iterator of (name, value) pairs, or of unique
        values representing (name, value) pairs."""
        raise NotImplementedError()

    def classify(self, features):
        """Given a set of features, return the most likely category for
        the sample and the probability of that category as predicted by the
        classifier's model of the sample space. Features should be passed
        in as an iterable or iterator of (name, value) pairs, or of unique
        values representing (name, value) pairs."""
        probabilities = self.probabilities(features)
        category = max(probabilities, key=probabilities.get)
        return category, probabilities[category]

    def train(self, observations):
        """Accept a sequence of multiple observations all at once."""
        for features, category in observations:
            self.observe(features, category)


class NBClassifier(Classifier):
    """This class implements the traditional Naive Bayes algorithm, except
    that it is left to the client to indicate not only the presence of a
    feature, but its absence. (Each value is in fact treated as a separate
    feature, whose absence is ignored.) This permits us to omit a feature
    if its value is unknown, and to assign more than just binary values to
    a feature."""

    def __init__(self, features=None, categories=None):
        self._priors = dict.fromkeys(categories or (), 0)
        self._conditionals = {feature: {} for feature in features or ()}

    def observe(self, features, category):
        """Update the model based on the features and category of the
        sample."""

        # If it's a previously unobserved category, add it to the table.
        if category not in self._priors:
            self._priors[category] = 0

        # Increment the appropriate counters in the tables.
        self._priors[category] += 1
        for feature in features:
            # If it's a previously unobserved feature, add it to the table.
            if feature not in self._conditionals:
                self._conditionals[feature] = {}
            self._conditionals[feature][category] = \
                self._conditionals[feature].get(category, 0) + 1

    def probabilities(self, features):
        """Return a dictionary mapping each category to its predicted
        probability for the sample."""

        # We will need to iterate over it more than once, so make sure that
        # we can.
        if not isinstance(features, (list, tuple)):
            features = list(features)

        for feature in features:
            # If it's a previously unobserved feature, add it to the table.
            if feature not in self._conditionals:
                self._conditionals = {}

        by_category = {}
        feature_count = len(self._conditionals)
        for category, category_count in self._priors.items():
            smoothed_count = category_count + feature_count
            proportion = category_count + 1
            for feature in features:
                proportion *= (
                    (self._conditionals[feature].get(category, 0) + 1) /
                    smoothed_count
                )
            by_category[category] = proportion

        total = sum(by_category.values())
        for category in by_category:
            by_category[category] /= total
        return by_category


class INBClassifier(Classifier):
    """This class implements a novel variant of the Naive Bayes algorithm,
    called Inertial Naive Bayes, which is designed to (asymptotically)
    eliminate oscillation of category predictions when the feature
    independence assumption of the Naive Bayes algorithm is violated. It
    does so by assigning greater weight to new observations which agree
    with the category currently predicted for a particular sample than to
    observations which disagree with the predicted category. Thus as new
    observations are made, the classifier tends to be wrong in the same
    cases as it was previously, unless it is capable of learning the
    correct classifications for the entire sample space. The primary
    utility of this variant of Naive Bayes is that stability in the
    predicted categories is a necessary quality of the subclassifiers used
    in the Less Naive Bayes algorithm."""

    def __init__(self, features=None, categories=None, bias=None):
        self._priors = dict.fromkeys(categories or (), 0)
        self._conditionals = {feature: {} for feature in features or ()}
        self._accuracy = .5
        self._observation_counter = 0

        self._bias = 1 if bias is None else bias

    def observe(self, features, category, weight=None):
        """Update the model based on the features and category of the
        sample."""

        if weight is None:
            weight = 1

        # We will need to iterate over it more than once, so make sure that
        # we can.
        if not isinstance(features, (list, tuple)):
            features = list(features)

        # If it's a previously unobserved category, add it to the table.
        if category not in self._priors:
            self._priors[category] = 0

        # The idea here is to reduce the update rate when the prediction
        # disagrees with the correct category, as compared to when they
        # agree, causing the predictions to stabilize even if the correct
        # mapping can't be learned.
        predicted_category, predicted_probability = self.classify(features)
        self._observation_counter += 1
        if predicted_category == category:
            accuracy_target = predicted_probability
            update_size = weight * (1 + self._bias)
        else:
            accuracy_target = 1 - predicted_probability
            update_size = weight
        self._accuracy += \
            (accuracy_target - self._accuracy) / self._observation_counter

        self._priors[category] += update_size
        for feature in features:
            self._conditionals[feature][category] = \
                self._conditionals[feature].get(category, 0) + update_size

    def probabilities(self, features):
        """Return a dictionary mapping each category to its predicted
        probability for the sample."""

        # We will need to iterate over it more than once, so make sure that
        # we can.
        if not isinstance(features, (list, tuple)):
            features = list(features)

        for feature in features:
            # If it's a previously unobserved feature, add it to the table.
            if feature not in self._conditionals:
                self._conditionals[feature] = {}

        by_category = {}
        feature_count = len(self._conditionals)
        for category, category_count in self._priors.items():
            smoothed_count = category_count + feature_count
            proportion = category_count + 1
            for feature in features:
                proportion *= (
                    (self._conditionals[feature].get(category, 0) + 1) /
                    smoothed_count
                )
            by_category[category] = proportion

        total = sum(by_category.values())
        for category in by_category:
            by_category[category] /= total
        return by_category

    def _get_bias(self):
        return self._bias

    def _set_bias(self, bias):
        assert bias >= 0
        self._bias = bias

    bias = property(
        _get_bias,
        _set_bias,
        doc="""The degree to which the classifier prefers to continue
        making the same mistakes when classification mistakes cannot be
        avoided."""
    )

    @property
    def accuracy(self):
        """A moving average of the predictive accuracy of the classifier
        as the model is developed."""
        return self._accuracy

    @property
    def conservative_accuracy(self):
        """A slightly more conservative estimate of the accuracy of the
        classifier which gives room for error due to insufficient
        observations."""
        return (
            self._accuracy *
            self._observation_counter / (1 + self._observation_counter)
        )

    @property
    def observation_counter(self):
        """The total number of training observations supplied to the
        classifier so far."""
        return self._observation_counter

    @property
    def categories(self):
        """The categories this classifier can return as predictions."""
        return frozenset(self._priors)


class LNBClassifier(Classifier):
    """This class implements a novel variant of the Naive Bayes algorithm,
    called Less Naive Bayes, which uses multiple INBClassifiers to
    iteratively reduce the sample space into linearly separable subspaces.
    Training is slow but estimation is fast. New subclassifiers are only
    added as needed. This class effectively learns the minimal
    representation of the problem. It will converge to the correct
    classifications, but not necessarily to the correct probabilities.
    """

    def __init__(self, features=None, categories=None, max_depth=None,
                 bias=None, look_ahead=10):
        assert max_depth is None or max_depth >= 1
        assert look_ahead >= 2

        self._features = frozenset(features or ())
        self._original_categories = frozenset(categories or ())

        self._most_accurate_layer = 0
        self._observation_counter = 0

        self._max_depth = max_depth

        self._bias = .01 if bias is None else bias
        self._look_ahead = look_ahead

        self._layers = [INBClassifier(self._features, bias=self._bias)]

    @property
    def bias(self):
        """The degree to which subclassifiers prefer to continue making
        the same mistakes when classification mistakes cannot be avoided.
        """
        return self._bias

    @property
    def depth(self):
        """The number of subclassifiers currently used to transform the
        problem space into a more separable one."""
        return len(self._layers)

    @property
    def best_depth(self):
        """The depth of the subclassifier with the most accurate
        predictions."""
        return self._most_accurate_layer

    @property
    def max_depth(self):
        """The maximum depth, or None if there is no maximum."""
        return self._max_depth

    def add_layer(self):
        """Increment the depth by adding another subclassifier. If the
        depth has reached the maximum, do nothing."""
        if self._max_depth is None or len(self._layers) < self._max_depth:
            self._layers.append(
                INBClassifier(self._features, bias=0)
            )

    def remove_layer(self):
        """Decrement the depth by removing the topmost subclassifier. If
        the depth has reached the minimum, do nothing."""
        if len(self._layers) > 1:
            self._layers.pop()

    def observe(self, features, category, weight=None):
        """Update the model based on the features and category of the
        sample."""

        # We will need to iterate over it more than once, so make sure that
        # we can.
        if not isinstance(features, (list, tuple)):
            features = list(features)

        original_category = category
        category = (original_category,)

        if original_category not in self._original_categories:
            self._original_categories |= frozenset([original_category])

        self._observation_counter += 1

        # Provide the new observation to each layer.
        for index, layer in enumerate(self._layers):
            if index >= self._most_accurate_layer - self._look_ahead:
                layer.observe(features, category, weight)
            if index + 1 >= len(self._layers):
                break
            predicted_category, probability = layer.classify(features)
            category = (predicted_category, original_category)

        # Only consider subclassifiers which have observed at least half
        # of the total number of observations provided so far.
        new_most_accurate_layer = max(
            (index for index in range(len(self._layers))
             if (not index or
                 (self._layers[index].observation_counter * 2 >=
                  self._observation_counter))),
            key=lambda index: self._layers[index].conservative_accuracy
        )

        if new_most_accurate_layer != self._most_accurate_layer:
            self._most_accurate_layer = new_most_accurate_layer
            for index, layer in enumerate(self._layers):
                if index <= self._most_accurate_layer:
                    layer.bias = self._bias / (index + 1)
                else:
                    layer.bias = 0

        most_accurate = self._layers[self._most_accurate_layer]
        last = self._layers[-1]

        upper_bound = len(self._layers) - self._look_ahead
        lower_bound = len(self._layers) - 2 * self._look_ahead

        if self._most_accurate_layer > upper_bound:
            max_observations = max(layer.observation_counter
                                   for layer in self._layers)
            if (self._most_accurate_layer == len(self._layers) - 1 or
                    last.observation_counter * 2 >= max_observations):
                self.add_layer()
        elif self._most_accurate_layer < lower_bound:
            if last.accuracy < most_accurate.conservative_accuracy:
                self.remove_layer()

    def probabilities(self, features):
        """Return a dictionary mapping each category to its predicted
        probability for the sample."""
        # Use the most accurate subclassifier to generate the predictions.
        layer = self._layers[self._most_accurate_layer]

        # Sum over predictions which include the same target category,
        # ignoring the components which only address predictions made by
        # earlier layers.
        results = {}
        for category, probability in layer.probabilities(features).items():
            target_category = category[-1]
            results[target_category] = \
                results.get(target_category, 0) + probability

        return results

    def get_values(self, function):
        """Return a list of values produced by passing each layer to the
        given function. Useful primarily for debugging."""

        return [function(layer) for layer in self._layers]


def make_ascii_plot(values, height=10):
    """Return a simple ASCII plot of the given values."""

    min_value = min(values)
    max_value = max(values)

    rows = ['+' + '-' * len(values) + '+']
    for index in range(height, 0, -1):
        threshold = min_value + (max_value - min_value) * index / height
        rows.append('|' + ''.join(
            ('*' if value >= threshold else ' ')
            for value in values
        ) + '|')
    rows.append('+' + '-' * len(values) + '+')

    return '\n'.join(rows)


class SampleGenerator:
    """A simple class to generate artificial samples according to a
    probability distribution for testing purposes."""

    def __init__(self, feature_names, value_factories, categorizer):
        self.feature_names = tuple(feature_names)
        self.value_factories = tuple(value_factories)
        self.categorizer = categorizer

        assert len(self.feature_names) == len(self.value_factories)

    def make_sample(self):
        """Create a new sample and return it."""
        values = [factory() for factory in self.value_factories]
        features = list(zip(self.feature_names, values))
        category = self.categorizer(*values)
        return features, category

    def __call__(self, count):
        """Create `count` new samples and return them in an iterator."""
        for _ in range(count):
            yield self.make_sample()


class RandomBooleanFunction:
    """A randomly generated probabilistic Boolean function, for testing."""

    def __init__(self, input_count):
        assert input_count >= 0
        self.input_count = input_count
        self.table = [
            random.random()
            for _ in range(1 << input_count)
        ]

    def __call__(self, *inputs):
        assert len(inputs) == self.input_count
        index = 0
        for value in inputs:
            assert value in (0, 1)
            index <<= 1
            index += value
        return random.random() <= self.table[index]


if __name__ == "__main__":
    import random
    import string
    import time

    input_count = 5
    even_probabilities = False
    number_of_samples = 50000

    if even_probabilities:
        value_factories = [lambda: bool(random.randrange(2))] * input_count
    else:
        value_factories = []
        for _ in range(input_count):
            probability = random.random()
            value_factory = lambda: random.random() <= probability
            value_factories.append(value_factory)

    categorizer = RandomBooleanFunction(input_count)

    sample_generator = SampleGenerator(
        feature_names=string.ascii_lowercase[:input_count],
        value_factories=value_factories,
        categorizer=categorizer
    )

    def show(classifier, target_function, run_time, individual=False):
        """Pretty-print the classifier's conclusions for the samples it
        was provided."""
        print("Classifier type:", type(classifier).__name__)
        if isinstance(classifier, (INBClassifier, LNBClassifier)):
            print("Bias:", classifier.bias)
        if isinstance(classifier, LNBClassifier):
            print("Depth:", classifier.depth)
            print("Best depth:", classifier.best_depth)
        print()

        correct_count = 0
        error = 0

        count = target_function.input_count

        for index in range(1 << count):
            target_probabilities = {
                True: target_function.table[index],
                False: 1 - target_function.table[index]
            }

            values = []
            for _ in range(count):
                values.insert(0, bool(index % 2))
                index >>= 1

            features = list(zip(string.ascii_lowercase[:count], values))

            probabilities = classifier.probabilities(features)

            for category in sorted(target_probabilities, reverse=True):
                target = target_probabilities[category]
                predicted = probabilities.get(category, 0)
                error += abs(target - predicted)

                if individual:
                    condition = ''.join(('~' if not value else '') + letter
                                        for letter, value in features)

                    print('P(' + str(category) + '|' + condition + ') =',
                          predicted, '(' + str(target) + ')')

            if individual:
                print()

            category = max(target_probabilities,
                           key=target_probabilities.get)
            if category == max(probabilities, key=probabilities.get):
                correct_count += 1

        percentage = str(round(correct_count / (1 << count) * 100, 1))
        print("Correct classifications:",
              correct_count, "of", 1 << count,
              '(' + percentage + ' %)')
        print("Average error:", error / (1 << count))
        print("Run time:", run_time, "seconds")
        print()

        if isinstance(classifier, LNBClassifier):
            values = classifier.get_values(
                lambda layer: layer.conservative_accuracy
            )
            print(make_ascii_plot(values))
            print("Accuracy by layer")
            print(min(values), '-', max(values))
            print()

            values = classifier.get_values(
                lambda layer: layer.observation_counter
            )
            print(make_ascii_plot(values))
            print("Age by layer")
            print(min(values), '-', max(values))
            print()

    nb_model = NBClassifier()
    unbiased_lnb_model = LNBClassifier(bias=0)
    slightly_biased_lnb_model = LNBClassifier(bias=.01)
    biased_lnb_model = LNBClassifier(bias=1)

    for model in (nb_model,
                  unbiased_lnb_model,
                  slightly_biased_lnb_model,
                  biased_lnb_model):
        start_time = time.time()
        model.train(sample_generator(number_of_samples))
        end_time = time.time()
        show(model, categorizer, end_time-start_time)
        print()
