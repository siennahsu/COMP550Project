from wordfreq import zipf_frequency
from nltk.corpus import wordnet as wn
#from pywsd.lesk import simple_lesk
from itertools import chain

threshold = 4


def word_replacement(line):
    sentence = line
    line = line.split(' ')

    result = []

    for word in line:
        #check if word is alphabetic, ignore words that are not - e.g. punctuation, numbers, etc.
        if word.isalpha():
            score = zipf_frequency(word, 'en')
            #print(f' word: {word} score: {score}')

            if score < threshold:
                

                synonyms = wn.synsets(word)
                lemmas = set(chain.from_iterable([w.lemma_names() for w in synonyms]))

                min_score = score
                best_replacement = word

                for lemma in lemmas:
                    if zipf_frequency(lemma, 'en') > min_score:
                        min_score = zipf_frequency(lemma, 'en')
                        best_replacement = lemma

                result.append(best_replacement)

            else:
                result.append(word)
        else:
            result.append(word)

    return result

if __name__ == '__main__':
    input_filename = 'normal_test.txt'
    output_file = 'simple_lex.txt'

    #result = word_replacement('His dissertation topic was `` Representation of American Sign Language for Machine Translation . ''')

    output = []

    with open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.rstrip()
            simplified = word_replacement(line)
            #print(simplified)
            simplified = " ".join(simplified)
            output.append(simplified)
    
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for line in output:
            output_file.write(line)
            output_file.write('\n')
            






