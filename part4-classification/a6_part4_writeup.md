# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?

Without the StandardScaler, the model does not perform well, as the features (age, salary, and gender) are on different scales. Age and salary have different ranges, and this can cause the logistic regression model to give more weight to one feature over another.

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.

With the StandardScaler, the model gives a higher accuracy of 85% since the features are scaled to have a mean of 0 and a standard deviation of 1. This is useful enough for predicting purchases based on age, salary, and gender, as long as the data used for training is accurate to the subject.

3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?

The model will struggle with predictions where the features (age, salary, and gender) do not align with the typical purchasing behavior learned during training. For instance, the model might have a pattern of underestimating purchases for certain age or salary ranges that are underrepresented in the training data

4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.

No. 