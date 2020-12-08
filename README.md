# *Automated Twitter Chatbot And Sentiment Analysis*
**Steven Zych - October 2020**

# Introduction

This project looks at **using real-time Twitter content to perform customer outreach.** In the subsequent sections, I will explain the process through which I **trained a neural network** to predict sentiment on tweets, built a **Twitter chatbot** with protocols for interacting with human users, and set up a **livestream** that combines all this functionality into one **automated process.**

In practice, this neural-network solution to customer outreach takes the brunt of the work out of human hands. By automating a sentiment-prediction system and letting a chatbot handle the beginnings of an interaction to gauge interest, human employees are free to spend their time on tasks a machine can't handle. **This saves time, energy, and money at the end of the day.**

> The data in this project was all sourced from Twitter by querying tweets that contained some spelling of `"amazon OR @amazon"`. Let it be clear that I have **no brand affiliation with Amazon.** I simply used tweets at and about Amazon because they're plentiful and polarized. Perfect for machine learning.

## Table Of Contents

The sections of this project are as follows:
1. Data Collection
1. Data Visualization
1. Neural Network
1. Twitter Chatbot
1. Twitter Stream
1. Conclusions
1. Business Recommendations
1. Future Work

## Libraries

The following libraries were used to make this project possible:
- Twython
- Keras
- NLTK
- Matplotlib
- Seaborn
- NumPy
- Pandas
- JSON
- Pickle
- Regular Expressions
- Smtplib

# Data Collection

## Data Collection

The data collection process was fairly simple. Using an account made via Twitter's [developer portal](https://developer.twitter.com/en), I made a series of time-staged API calls that returned tweets meeting my specifications. Just over **10,000 tweets** were gathered. (Twitter limits free accounts to 50,000 per month, and I couldn't feasibly label enough tweets in a set bigger than 10,000 to give my semi-structured learning any validity.)

The loop below works by calling a **`Twython` object** (which is mostly there to store my API keys), and using its `.search()` method. This method is set to return 150 full tweets per iteration, only in English, and up until the `last_id`. `last_id` is set to re-assign on every iteration to whatever the last tweet's unique ID was in the previous call. This ensures that **no tweets are gotten twice,** since Twitter's search functionality defaults to just return the most recent tweets that fulfill a request.

```
last_id = None

query = '@amazon OR amazon OR Amazon -filter:retweets'

for i in range(67):
    search = t.search(q = query,
                      max_id = last_id,
                      lang = 'en',
                      tweet_mode = 'extended',
                      count = 150)
    with open(f'data_no_RT/data_{i}.txt', 'w') as outfile:
        json.dump(search, outfile)
    last_id = search['statuses'][-1]['id']
    time.sleep(310)
```

## Data Labeling

I opted for a **semi-structured approach** to machine learning for this project. This required that a decent chunk of the data be labeled before machine learning. I set up a function `label_sentiment()` (found in `data_labeling.ipynb`) which passed me random tweets from the collection one-at-a-time. The function then asked for an input which was saved to `/labels/labels.txt`. **I manually labeled 2,014 of the 10,164 tweets,** about 20%. The distribution of labels by tweet showed the following:

![Start Tweet Labels](/images/count_initial.png)

The results in the next plot wouldn't be visible for a long while to me, but so that you can see them together, this is how the same distribution looked **after all the tweets had been labeled by the neural network:**

![End Tweet Labels](/images/count_final.png)

> There is a minor increase in the percentage of tweets labeled `Positive`, but not enough to indicate that the neural network shows strong misclassification.

With a labeled set of tweets, I was ready to prep them to model.

## Data Cleaning

**Cleaning this data,** fortunately, was straightforward as well. Punctuation, usernames, URL's, and hashtags were removed from the data, as were stopwords. Emojis were kept in because they carry so much emotional value. Each tweet was tokenized and stemmed, producing cleaning changes such that this:

`"@username I just meant the amazon link friend. 😭😭😭 TY!"`

Would become a list like this:

`["meant", "amazon", "link", "friend", "😭😭😭", "ty"]`

I then used the **NLTK Tokenizer** to turn these lists into **padded sequences** of integers so that they could be read as neural network inputs. An example (not of the tweet above) takes the form of a NumPy array like this, where **each unique word in the `total_vocabulary` has been given a numerical value.** The abundance of zeroes allows for tweets of up-to-forty-five words to be used as inputs--they're all the same length now.

`array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0, 3686, 4353,  970,  122, 2378, 1054, 2609,   49,
        306])`

These 2014 sequences, along with their labeled sentiments, were fed into an initial neural network. But before we touch on that, I'd like to share a visualization of the **25 most common bigrams** (two words that appear adjacent) in the data. This plot highlights another one of the challenges with handling this data--two kinds of Amazon:

![25 Bigrams](/images/bigram_plot.PNG)

It was impossible to filter for tweets about Amazon the company but not ***the*** Amazon at the same time. You'll notice that bigrams 8 through 15 (from "tip point" to "savannah -") are about the rainforest, not the tech giant. That being said, the fact that there are many **bigrams** about the Amazon rainforest does not necessarily mean that there was an equally large amount of **tweets** about the Amazon rainforest in the data, but rather that people talking about the Amazon rainforest show more homogeneity in their vocabulary than people talking about Amazon the brand. With that out of the way, we move on to the neural network.

# Neural Network

The baseline model used an **architecture** that took the general shape of:
- Embedding layer based on max tweet length and `total_vocabulary`
- LSTM layer to handle data occurring in sequence
- Pooling layer to reduce complexity
- Dropout to reduce overfitting
- Dense layer with 25 nodes and L2 regularization
- Output layer with 3 nodes for the sentiments `Negative`, `Neutral`, and `Positive`

This model performed decently well, considering the exceptionally small dataset it was provided. (Again, the smallness of this data was required by my time and labor-cost constraints in doing this as a capstone. Just one person in a couple weeks can't spend all that time labeling data.) Numerous alternate versions were tried for a baseline, but overfitting proved to be somewhat unavoidable.

**As models were trained, another set of tweets would have labels put on them,** and a new neural network would be made. The first batch of 2,014 tweets produced a neural network that predicted labels on the next 1,000 tweets. These 3,014 labeled tweets made a second NN which predicted labels on 1,500 *more* tweets. **This pattern repeated** until all the tweets were labeled and a final model `stem_model_5` was produced.

A final **validation accuracy of 73%** was achieved. Considering the **extreme** diversity of content coming in (Twitter users can output any string imaginable as a tweet), I don't think this is too bad.

![Model Performance](/images/model.PNG)

# Twitter Chatbot

This is where things get cool. If you want the lengthy rundown, head on over to `twitter_chatbot.ipynb` as I'll just be providing a summary here. 

Through the process of defining five key functions `send_message()`, `store_messages()`, `clean_message()`, `greet()`, and `reply()` I built **the class `Conversation`,** which represents the unique interaction between one Twitter user and the chatbot. I'll be explaining each of these functions briefly before moving on to show how the `Conversation` class works within the Twitter livestream.

## `Conversation` Class

This class takes two inputs: A `Twython` object and a `user_id`. The `Twython` object contains all of our authentication information and the `user_id` is the unique ID of the human user we're going to chat with. Instantiating looks like:

```
c = Conversation(t = Twython, user_id = 1234567890)
```

Once we've done this, all the following functions are available to us.

## `send_message()`

The most important function for a chatbot to have, `send_message()` puts a cleaner wrapper on `Twython.send_direct_message()` which normally looks like this:

```
t.send_direct_message(event = {"type" : "message_create",
                               "message_create" :
                                   {"target": {"recipient_id" : USER_ID},
                                   "message_data" : {"text" : text}}}
)
```

The clean version looks like this:

```
c.send_message(text = "This is a test of the send_message() function",
              user_id = 1234567890,
              t = t)
```

And produces:

![Send Message](/images/reply_send_message_function.PNG)

## `store_messages()`

Now that we can send messages, it's imperative that we can *store* the messages coming in. When we call `store_messages()` and then reference the `Conversation` object's `messages` attribute we get something like this:

```
[{'time': '1602973368365',
  'user_id': '1234567890',
  'text': 'oh gotcha. yes i wanna collaborate!'},
 {'time': '1602973242698',
  'user_id': '1234567890',
  'text': 'pls help me dood'},
 {'time': '1602972588876',
  'user_id': '1234567890',
  'text': 'Yes test'}]
```

A succinct transcript of everything the human user has sent the bot sits above. We can now reference the most recent message when we want to reply.

## `clean_message()`

But there's one more piece of the puzzle before we can reply! Remember the cleaning from before? Nearly the **same** cleaning process that was used to prep data for the neural network is used here, with the additional step of pulling out the words `['yes', 'no', 'help, 'stop']` for hardcoded responses. For example, the message `"hahaha cool computer, yes i'll work with u!"` becomes simply `['yes']`. A response like this is much easier to answer.

## `reply()`

This function relies on all of those before it within this class, and does the brunt of the work. In short, it has **planned responses** to the four possible inputs offered above, and extra functionality to address messages that are too short, too long, contain multiple command words, *or* need a human to step in. Moreover, it has the ability to update the `Conversation` class's previously-unmentioned attributes `help_count` and `no_contact`. The former tracks how many times someone asks the bot for `'help'` and ends the convo early if it hits 3 or more. The latter allows for users to end the convo, and breaks all messaging functions once set to `True`.

An example of using reply can look something like this:

![Convo](/images/convo_example.PNG)

Which in the background is doing this:

```
c.store_messages()
newest_message = c.messages[0]['text']
c.reply(newest_message)
```

## `greet()`

Lastly, function greets the user--it's as simple as that. A message is sent that asks whether or not they'd like to work with the brand in question--Amazon in this case:

![Greet](/images/reply_greet_amazon.PNG)

This works by simply calling the earlier-defined function `send_message()` with pre-planned content.


# Twitter Streaming

In order for that bot to be *useful,* it needs to be contacting *real* users, in *real* time. This is precisely what the **class `Stream`** does. This class is explored more thoroughly in the notebook `twitter_streaming.ipynb`. Visually, it works like this:

![Streaming Flowchart](/images/icon_flowchart.jpg)

## `Stream` Class

This class inherits from the parent `TwythonStreamer` and is loosely based on the default custom streamer provided in the [Twython documentation](https://twython.readthedocs.io/en/latest/usage/streaming_api.html)--the `on_error()` function is the same, but nothing else is. The `on_success()` method I've defined is what **brings everything together.** Let's print out all its glory with some explanation:

```
class Stream(TwythonStreamer):

    def on_success(self, data):
# Check for enough clout
        if data['user']['followers_count'] > 1000:
# Check for positive sentiment
            if tweet_sentiment(data['text']) == 3:
# Instantiate Emailer
                emailer = Emailer(sender_email = "user@company.com",
                                  password = "password")
# Instantiate Conversation
                c = Conversation(user_id = data['id'], t = t)
# Greet user
                c.greet()
# Converse with user until they opt out or collaborate
                while (c.no_contact == False) and (c.passed_to_human == False):
                    time.sleep(120)
                    c.store_messages()
                    if c.messages != None:
                        newest_message = c.messages[0]['text']
                        c.reply(newest_message)
# Send an email to the employee tasked with chat
                if c.passed_to_human == True:
                    emailer.assign_chat(rec_email = "employee@company.com",
                                        user_id = c.user_id)
```

Here's the same thing back **line-for-line** in a human-readable way:
- If the Twitter stream connection works,
- and if the tweet's user has at least 10,000 followers,
- *and* if the sentiment of their tweet about Amazon is positive,
- then start a conversation with them.
- Greet them.
- So long as they haven't opted out, and the bot hasn't got a human on the job,
- wait two minutes (Twitter's DM database take a moment to update),
- and store the messages they've sent you.
- If they *have* sent a message,
- save the newest one,
- and reply to it!

The portion of that process under `# Converse ...` repeats so long as the human will participate, and so long as those two conditions aren't broken. Now, if a user we're conversing with *does* choose to collaborate with the brand in question: We break from this loop, and an email is sent to a human employee to finish out the deal. It's imperative that the chatbot is not responsible for making monetary offers.

Whenever a `Conversation` ends, we go back to streaming. Once a human initializes the `Stream`, **all of this is automated.**

# Conclusions

## Business Recommendations

The way I see it this is a **modular business solution.** As the `Stream` class stands in the example above, we're contacting Twitter users who tweeted **positively** about **Amazon** right **now** who have at least **10,000** followers. All of these variables are just that: Variables. We could instead reach out to **neutral** users who were talking about **Wizards Of The Coast** last **week** with at least **5,000** followers and try and sway their opinion one way or the other. In other words, this proof of concept is a **tool** more than it is a singular solution. I've even gone on to think up a couple more possible implementations beyond this:

1. Reach out exclusively to users tweeting **negatively** about your brand to initiate customer service. This only requires a change in the `greet()` function's text.
1. Collect **data on how users respond to the chatbot** to refine its tactics.
1. Use the livestream of tweets to **refine the neural network's** prediction ability.

## Future Work

What's most exciting about this bot is the prospect of **growth.** There's a handful of technical features (as well as more practical considerations) that will greatly improve not only the bot's performance, but the neural net's as well. And, as I mentioned before, this is a capstone project. There are a few things I wanted to implement but simply didn't have the time to. As of today, October 19th 2020, I plan to implement the following:

- **Neural network support on chatbot replies,** to allows for both hardcored `['yes, 'no', 'help', 'stop']` replies as well as more "chatting."
- **Increased amount of manually-labeled data.** This will make the neural net and the stream perform *way* better for predictions.
- Functionality for the **`Stream` to send emails** and distribute workload to different employees for following up on interested users.
- Research **asynchronous execution and observables in Python** to see how I can get multiple `Stream` and/or `Conversation` objects running at once.

## Thank You

I'd like to thank my instructor at Flatiron School Abhineet Kulkarni for being a kind and patient teacher, my online cohort for enduring support, and the thousands of faceless Twitter users whose data made the entire project possible. I'd also like to thank Twitter for keeping their data freely accessible, and Ryan McGrath for maintaining Twython and saving me from a lifetime of headache.