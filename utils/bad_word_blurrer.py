from better_profanity import profanity

profanity.load_censor_words()
BAD_WORDS = [str(w).lower() for w in profanity.CENSOR_WORDSET]

def blur_text(text):
    
    words = text.split()
    censored = []

    for word in words:
        clean = word.lower().strip(".,!?()[]{}:;\"'") 
        if clean in BAD_WORDS and len(clean) > 2:
            censored_word = word[0] + '*' * (len(word) - 2) + word[-1]
            censored.append(censored_word)
        else:
            censored.append(word)

    return ' '.join(censored)


