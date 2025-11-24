from blingfire import text_to_words
from blingfire_utils import BlingFireUtils

text = "Hello world. This is a test."

sentences = BlingFireUtils.GetSentencesWithOffsets(text)
print(sentences)