def permute (word, list, removed):

    for i in word :
        if i != '_' and i not in removed:
            removed.append(i)
            stripped_word = word.copy()
            for j in range(len(word)) :
                if stripped_word[j] == i :
                    stripped_word[j] = '_'

            list.append(stripped_word)
            list = permute(stripped_word, list, removed.copy())
    
    return list
    

filename = "words_250000_train.txt"


with open(filename,'r') as f:
    content = f.read()
    words = content.split()
    
print(f"{len(words)} words")
data = "strip.txt"
batch_size = 100
with open(data, 'w') as f:
    for batch in range(len(words)//batch_size): 
        for word in words[batch_size*batch:(batch_size + 1)*batch]:
            chunk = ''
            word_as_list = []
            for i in word: word_as_list.append(i)
            strip_list = permute(word_as_list, [], [])
            for strip_word_as_list in strip_list:
                strip_word = ''
                for i in strip_word_as_list : strip_word += i
                chunk += strip_word + " " + word + '\n'
            f.write(chunk)
            chunk = ''
        print(f"{batch} batches complete")