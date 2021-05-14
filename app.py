from pywebio.input import input, input_group, TEXT, NUMBER, FLOAT, actions
from pywebio.output import put_markdown, use_scope, clear
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
	put_markdown('# Writer Buddy')

	try_another = True

	while True:
		if try_another:
			clear('B')

			inputs = input_group('Inputs', [
					  input('Enter the first few words', value='Once upon a time, ', type=TEXT, name='title'),
					  input('Enter the number of characters in the article', value=100, type=NUMBER, name='length')
					  ])

			with use_scope('A'):
				put_markdown("**Writing the article. Come back in a few minutes...**")

			tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
			model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

			if inputs['title']:
				input_ids = tokenizer.encode(inputs['title'], return_tensors='pt')

				output = model.generate(input_ids, max_length=inputs['length'], do_sample=True, temperature=1)
										# num_beams=8, no_repeat_ngram_size=2, early_stopping=True

				text = tokenizer.decode(output[0], skip_special_tokens=True)

				clear('A')

				with use_scope('B'):
					put_markdown(text)

				try_another = actions(label="Would you like to try another?", 
									  buttons=[{'label': 'Yes', 'value': True}, 
											   {'label':'No', 'value': False}])

		else:
			clear('B')
			put_markdown('Thanks for trying out Writer Buddy.')
			break
			
if __name__ == '__main__':
    import argparse
    from pywebio.platform.tornado_http import start_server

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(main, port=args.port)
