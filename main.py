from tqdm import tqdm
from deep_translator import GoogleTranslator
import nltk.data
from haystack.nodes import QuestionGenerator
from datetime import datetime
import re

class FineTunePrepapred:
  def __init__(self):
    self.source = 'auto'
    self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    self.question_generator = QuestionGenerator(model_name_or_path="valhalla/t5-base-e2e-qg")

  def translator(self, target, text):
    return GoogleTranslator(source=self.source, target=target).translate(text)

  def tokenize(self, text):
    return self.tokenizer.tokenize(text)

  def save(self, output):
    print("save")
    now = datetime.now()
    with open(f"fine_tune_instruction_{now.strftime('%d-%m-%Y-%H:%M:%S')}.jsonl", 'w') as file:
      file.writelines("{\"prompt\": \"%s\", \"completion\": \"%s\"}\n" % (row['prompt'], row['completion']) for row in output)

  def prepare(self, sentences):
    translate_tag = "<-1--99043593--1->"
    output = []
    for i in tqdm(range(len(sentences))):
      try:
        questions = self.question_generator.generate(str(sentences[i]))
      except:
        continue

      for question in questions:
        translate = self.translator("ru", f"{question}{str(translate_tag)}{sentences[i]}")
        pc = translate.split(str(translate_tag))
        prompt = re.sub('"', "'", pc[0])
        completion = re.sub('"', "'", pc[1])
        output.append({'prompt': prompt, 'completion': completion})

    self.save(output)

INPUT_TEXT_FILE = "aggression_adolescents_with_general_giftedness.txt"

def main():
  with open(INPUT_TEXT_FILE, 'r') as file:
    text = file.read()
    ftp = FineTunePrepapred()
    en_text = ftp.translator('en', text)
    sentences = ftp.tokenize(en_text)
    ftp.prepare(sentences)

main()