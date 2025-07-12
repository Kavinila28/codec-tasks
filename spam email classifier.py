messages = [
    "Win a prize", 
    "Team lunch today", 
    "Free vacation deal", 
    "Project update meeting"
]

labels = [1, 0, 1, 0]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
data = vect.fit_transform(messages)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(data, labels)

test = ["Congrats, you’ve won free tickets"]
test_data = vect.transform(test)
is_spam = model.predict(test_data)[0]

if is_spam:
    print(messages,"\n Looks like SPAM!")
else:
    print("Looks like not a SPAM!")
