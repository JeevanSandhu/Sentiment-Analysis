class SentimentIntensityAnalyzer(object):
    def __init__(self, lexicon_file="vader_lexicon.txt"):
        self.lexicon_file = nltk.data.load(lexicon_file)
        self.lexicon = self.make_lex_dict()

    def _sift_sentiment_scores(self, sentiments):
            # want separate positive versus negative sentiment scores
            pos_sum = 0.0
            neg_sum = 0.0
            neu_count = 0
            for sentiment_score in sentiments:
                if sentiment_score > 0:
                    pos_sum += (float(sentiment_score) +1) # compensates for neutral words that are counted as 1
                if sentiment_score < 0:
                    neg_sum += (float(sentiment_score) -1) # when used with math.fabs(), compensates for neutrals
                if sentiment_score == 0:
                    neu_count += 1
            return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
            if sentiments:
                sum_s = float(sum(sentiments))
                # compute and add emphasis from punctuation in text
                punct_emph_amplifier = self._punctuation_emphasis(sum_s, text)
                if sum_s > 0:
                    sum_s += punct_emph_amplifier
                elif  sum_s < 0:
                    sum_s -= punct_emph_amplifier

                compound = normalize(sum_s)
                # discriminate between positive, negative and neutral sentiment scores
                pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

                if pos_sum > math.fabs(neg_sum):
                    pos_sum += (punct_emph_amplifier)
                elif pos_sum < math.fabs(neg_sum):
                    neg_sum -= (punct_emph_amplifier)

                total = pos_sum + math.fabs(neg_sum) + neu_count
                pos = math.fabs(pos_sum / total)
                neg = math.fabs(neg_sum / total)
                neu = math.fabs(neu_count / total)

            else:
                compound = 0.0
                pos = 0.0
                neg = 0.0
                neu = 0.0

            sentiment_dict = \
                {"neg" : round(neg, 3),
                 "neu" : round(neu, 3),
                 "pos" : round(pos, 3),
                 "compound" : round(compound, 4)}

            return sentiment_dict


                    def polarity_scores(self, text):
            """
            Return a float for sentiment strength based on the input text.
            Positive values are positive valence, negative value are negative
            valence.
            """
            sentitext = SentiText(text)
            #text, words_and_emoticons, is_cap_diff = self.preprocess(text)

            sentiments = []
            words_and_emoticons = sentitext.words_and_emoticons
            for item in words_and_emoticons:
                valence = 0
                i = words_and_emoticons.index(item)
                if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and \
                    words_and_emoticons[i+1].lower() == "of") or \
                    item.lower() in BOOSTER_DICT:
                    sentiments.append(valence)
                    continue

                sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

            sentiments = self._but_check(words_and_emoticons, sentiments)

            return self.score_valence(sentiments, text)