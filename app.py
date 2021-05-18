from pywebio.input import input, input_group, TEXT, NUMBER, FLOAT, actions, radio
from pywebio.output import put_markdown, use_scope, clear
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import argparse
from pywebio import start_server

def app():
	put_markdown('# Writer Buddy')
	try_another = True

	while True:
		if try_another:
			clear('B')

			inputs = input_group('Inputs', [
					  input('Enter the first few words', value='Once upon a time, ', type=TEXT, name='title'),
					  input('Enter the number of characters in the article', value=100, type=NUMBER, name='length'),
					  radio("Which model do you want to use to generate the text? (Bigger model => longer time)", options=['Distil GPT2', 'GPT2 Large', 'GPT3 Neo'], required=True, name='model')
					  ])

			with use_scope('A'):
				put_markdown("**Writing the article. Come back in a few minutes...**")

			if inputs['model']=='Distil GPT2':
				tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
				model = GPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

				if inputs['title']:
					input_ids = tokenizer.encode(inputs['title'], return_tensors='pt')

					output = model.generate(input_ids, max_length=inputs['length'], do_sample=True, temperature=0.8, no_repeat_ngram_size=2, early_stopping=False)

					text = tokenizer.decode(output[0], skip_special_tokens=True)

			elif inputs['model']=='GPT2 Large':
				tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
				model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

				if inputs['title']:
					input_ids = tokenizer.encode(inputs['title'], return_tensors='pt')

					output = model.generate(input_ids, max_length=inputs['length'], do_sample=True, temperature=1)
											# num_beams=8, no_repeat_ngram_size=2, early_stopping=True

					text = tokenizer.decode(output[0], skip_special_tokens=True)

			elif inputs['model']=='GPT3 Neo':
				generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
				res = generator(inputs['title'], max_length=inputs['length'], do_sample=True, temperature=0.9)

				text = res[0]['generated_text']

			clear('A')

			with use_scope('B'):
				put_markdown(text)

			outputs = input_group('', [
					actions(label='', buttons=[{'label': 'Try another article', 'value': True}], name='try_another'),
					input("Want to save it?", value=inputs['title']+'.txt', name='save_location')
				    ])

			if outputs['save_location']:
				with open('Genereted Text Samples/{}.txt'.format(outputs['save_location'].split('txt')[0]), 'w') as f:
					f.write(text)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(predict, port=args.port)
