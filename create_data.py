def permute(word, results, removed):
    for i in word:
        if i != '_' and i not in removed:
            removed.append(i)
            stripped_word = word.copy()
            for j in range(len(stripped_word)):
                if stripped_word[j] == i:
                    stripped_word[j] = '_'
            results.append(stripped_word)
            permute(stripped_word, results, removed.copy())
    return results


filename = "words_250000_train.txt"

with open(filename, 'r') as f:
    words = f.read().split()

print(f"{len(words)} words")

output_file = "small_strip_250000.txt"
batch_size = 1000

with open(output_file, 'w') as f:
    for batch in range(len(words) // batch_size):
        for word in words[batch_size * batch : (batch + 1) * batch_size]:
            word_as_list = list(word)
            strip_list = permute(word_as_list, [], [])
            for strip_word_as_list in strip_list:
                strip_word = ''.join(strip_word_as_list)
                f.write(f"{strip_word} {word}\n")
        if (batch % 100 == 0) :
            print(f"{batch + 1} batches complete of out {len(words) // batch_size}")
