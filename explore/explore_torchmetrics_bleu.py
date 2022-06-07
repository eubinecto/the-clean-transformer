from torchmetrics import functional as F  # noqa


# reference corpus must be an iterable (instances) of iterables (candidates) of iterables (tokens) of strings
answers = [["I love you more than anyone else".split()], ["I hate you so much".split()]]
# translation corpus must be an iterable (instances) of iterables (tokens) of strings
predictions = ["I love you so much".split(), "I hate you so much".split()]

# the scores should decrease as n_gram increases because the window of exact matches increases
print(F.bleu_score(answers, predictions, n_gram=1))
print(F.bleu_score(answers, predictions, n_gram=2))
print(F.bleu_score(answers, predictions, n_gram=3))
print(F.bleu_score(answers, predictions, n_gram=4))  # default
print(F.bleu_score(answers, predictions, n_gram=5))
# this should be zero because no predictions have length longer than 6
# it is impossible to have 6-ngram matches (exact matches of length 6)
print(F.bleu_score(answers, predictions, n_gram=6))
print("after smoothing is applied")
# why apply smoothing?: https://aclanthology.org/P04-1077.pdf
print(F.bleu_score(answers, predictions, n_gram=1, smooth=True))
print(F.bleu_score(answers, predictions, n_gram=2, smooth=True))
print(F.bleu_score(answers, predictions, n_gram=3, smooth=True))
print(F.bleu_score(answers, predictions, n_gram=4, smooth=True))
print(F.bleu_score(answers, predictions, n_gram=5, smooth=True))
print(F.bleu_score(answers, predictions, n_gram=6, smooth=True))
