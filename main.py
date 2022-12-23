from nltk.corpus import stopwords
import math
import nltk
nltk.download('stopwords')
nltk.download("punkt")

#######################################################
# Tokenization
#######################################################

tokonize = []
filterd = []
stopwords_list = ['.', ',', "&", "|"]
files = ["1.txt", "2.txt", "3.txt","4.txt", "5.txt", "6.txt","7.txt", "8.txt", "9.txt", "10.txt"]
for x in files:
    f = open(x, "r")
    tokonize += nltk.word_tokenize(f.read())

#######################################################
# Remove Stop Words
#######################################################


stopwords = set(stopwords.words('english'))
stopwords.remove('in')
stopwords.remove('to')
stopwords.remove('where')

# print(stopwords_list)
for j in tokonize:
    for y in stopwords_list:
        if j == y:
            tokonize.remove(j)


#######################################################
# Make the words lower Case and Sort them
#######################################################

tokonize = [word.lower() for word in tokonize]

tokonize = sorted(tokonize)
tokonize = list(dict.fromkeys(tokonize))
for i, word in enumerate(tokonize):
    if word[len(word)-1] == '.' or word[len(word)-1] == ',':
        tokonize[i] = word[:-1]

# {hello : {"1.txt": [21, 21 , 262], "2.txt": [232, 3333]}, skn: {"4.txt": [121]}}

#######################################################
# Positional index model
#######################################################
# {term: {file: [15, 20, ...], file: [...], file: [222]}, term2: [...]}, term: {file: [21]}}
positional_index = {}

for word in tokonize:
    for file in files:
        f = open(file, "r")
        file_content = f.read()
        file_content_list = file_content.split()
        file_content_list = [word.lower() for word in file_content_list]
        for index, file_word in enumerate(file_content_list):
            if file_word[-1] == '.' or file_word[-1] == ',':
                file_word = file_word[:-1]
            index += 1
            if word == file_word:
                if len(positional_index) == 0:
                    # sett = {word: {file: [index]}}
                    positional_index[word] = {file: [index]}
                else:
                    if word in positional_index:
                        if file in positional_index[word]:
                            positional_index[word][file].append(index)
                        else:
                            positional_index[word][file] = [index]
                    else:
                        positional_index[word] = {file: [index]}

#######################################################
# Print The Constructing Auxiliary structure
#######################################################


for word in positional_index:
    length = 0
    for file in positional_index[word]:
        length += len(positional_index[word][file])
    print("< " + word + " " + str(length))
    for file in positional_index[word]:
        indexes = ', '.join(map(str, positional_index[word][file]))
        print(file + ": " + indexes + ";")
    print()


#######################################################
# Phase Query and print solution
#######################################################

query = input("Enter Your Query: ")
query_tokonization = nltk.word_tokenize(query)
query_tokonization = [word.lower() for word in query_tokonization]
for j in query_tokonization:
    for y in stopwords_list:
        if j == y:
            query_tokonization.remove(j)
# look like chaineeses
solutions = {}
print("Found IN:")
flag = False
for i in range(1, len(query_tokonization)):
    if query_tokonization[i] in positional_index and query_tokonization[i-1] in positional_index:
        for file_name in positional_index[query_tokonization[i-1]]:
            if file_name in positional_index[query_tokonization[i]]:
                for index in positional_index[query_tokonization[i-1]][file_name]:
                    if index+1 in positional_index[query_tokonization[i]][file_name]:
                        if flag == False:
                            if file_name in solutions:
                                if index not in solutions[file_name]:
                                    solutions[file_name].append(index)
                            else:
                                solutions[file_name] = [index]
                    else:
                        if flag == True:
                            if index-i+1 in solutions[file_name]:
                                solutions[file_name].remove(index-i+1)
            else:
                if flag == True:
                    if file_name in solutions:
                        solutions.pop(file_name)

    else:
        if flag == True:
            solutions = {}
    flag = True


for file in solutions:
    indexes = ', '.join(map(str, solutions[file]))
    print(file + " " + indexes)


#######################################################
# Vector space model
#######################################################

# {file2: {term1: [idf, tf, tf weight, tf.idf, normalize], term2: [idf, tf, tf weight, tf.idf, normalize]}, file2: {...}}
vector_space_model = {}

for file in files:
    vector_space_model[file] = {}
    for term in positional_index:
        if file in positional_index[term]:
            idf = math.log(len(files) / len(positional_index[term]))
            tf = len(positional_index[term][file])
            tf_weight = 1 + math.log(tf)
            tf_idf = tf_weight * idf
            vector_space_model[file][term] = [idf, tf, tf_weight, tf_idf]
        else:
            vector_space_model[file][term] = [0, 0, 0, 0]

# add the normalize of every term
for file in vector_space_model:
    tf_idf_square_summation = 0
    for term in vector_space_model[file]:
        tf_idf_square_summation += vector_space_model[file][term][3] * \
            vector_space_model[file][term][3]
    length_of_document = math.sqrt(tf_idf_square_summation)
    for term in vector_space_model[file]:
        if sum(vector_space_model[file][term]) == 0:
            vector_space_model[file][term].append(0)
        else:
            vector_space_model[file][term].append(
                vector_space_model[file][term][3]/length_of_document)

# print(vector_space_model)

query_vector_space_model = {}


for word in query_tokonization:
    idf = math.log(len(files) / len(positional_index[term]))
    tf = 0
    for qeury_word in query_tokonization:
        tf += 1
    tf_weight = 1 + math.log(tf)
    tf_idf = tf_weight * idf
    query_vector_space_model[word] = [idf, tf, tf_weight, tf_idf]

for word in query_vector_space_model:
    tf_idf_square_summation = 0
    for qeury_word in query_vector_space_model:
        tf_idf_square_summation += query_vector_space_model[qeury_word][3] * \
            query_vector_space_model[qeury_word][3]
    length_of_query = math.sqrt(tf_idf_square_summation)
    query_vector_space_model[word].append(
        query_vector_space_model[word][3]/length_of_query)

#######################################################
# Print Similarity of query with doc.
#######################################################
for file in files:
    similarity = 0
    for word in query_vector_space_model:
        if word in vector_space_model[file]:
            similarity += query_vector_space_model[word][4] * \
                vector_space_model[file][word][4]
    print("Similarity between query and document " +
          file + ": " + str(similarity))
