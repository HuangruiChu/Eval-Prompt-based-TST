

def gen_prompt(input_sentence, to_style ,prompt_style="zero_shoot"):
  '''
  generate the prompt for the specific sentence
  
  # label should be in ["neg", "pos"]
  
  '''

  if to_style == "pos":
    to_style = "negative"
  elif to_style == "neg":
    to_style = "positive"
  
  if prompt_style == "zero_shoot":
    answer = '''Here is some text: \"{}\". Here is a rewrite of the text, which is more {}: '''.format(input_sentence,to_style)
  elif prompt_style == "few_shoot":
    answer = '''Here is some text: "I was really sad about the loss". Here is a rewrite of the text, which is more positive: "I was able to accept and work through the loss to move on." Here is some text: "The eggnog was tasteless". Here is a rewrite of the text, which is more positive: "The eggnog had a great, festive taste to it."Here is some text: \"{}\". Here is a rewrite of the text, which is more {}: '''.format(input_sentence,to_style)
  
  elif prompt_style == "augment_shoot":
    answer = '''Here is some text: "When the doctor asked Linda to take the medicine, he smiled and gave her a lollipop". Here is a rewrite of the text, which is more scary: "When the doctor told Linda to take the medicine, there had been a malicious gleam in her eye that Linda didn't like at all" Here is some text: "They asked loudly, over the sound of the train". Here is a rewrite of the text, which is more intense: "They yelled aggressively, over the clanging of the train" Here is some text:  \"{}\". Here is a rewrite of the text, which is more {}:'''.format(input_sentence,to_style)
  elif prompt_style == "paraphrase":
    answer = '''Here is some text: \"{}\". Here is a paraphrase of the text, which removes the style but preserves the content:'''.format(input_sentence)
  else:
    answer = '''Rewrite to be more {}:\n{}'''.format(to_style, input_sentence)
  return answer
