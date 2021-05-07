from transformers import GPT2LMHeadModel, GPT2Tokenizer
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('Writer buddy', external_stylesheets=external_stylesheets)
server = app.server

colors = {
	'background': '#ffffff',
	'text': '#33B5FF'
}

app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'height':'100vh', 'width':'100%', 'height':'100%', 'top':'0px', 'left':'0px'}, 
	children=[
		html.H1(children='Writer Buddy Lite'),
		
		dcc.Input(id='input_sentence', placeholder='Enter the first few words', type='text', style={'width': '50%'}),
		html.Div([
			dcc.Input(id='input_length', placeholder='Enter the length of the article (default - 100 characters)', type='text', style={'width': '25%'}),
		]),

		html.Button('Submit', id='submit_btn', style={"margin-top": "15px"}),
		html.H6(id='load', style={"margin-top": "25px"}),
		html.H6(id='article_display', style={'height':'6vh', 'margin-top': '25px', 'font-size':'1.15em'}),
	])

@app.callback(Output('load', 'children'),
              [Input('submit_btn', 'n_clicks')],
              [State('input_sentence', 'value')])
def prepare_data(input_sentence, submit_btn):
	if submit_btn:
		if input_sentence=='':
			return dash.no_update
		else:
			return html.Div([dcc.Markdown(
				'''Writing the article, come back in a few minutes...''')], id='article_display')

@app.callback(
	Output('article_display', 'children'),
	[Input('submit_btn', 'n_clicks')],
	[State('input_sentence', 'value'),
	 State('input_length', 'value')])
def generate_article(n_clicks, input_sentence, input_length):
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
	
	if n_clicks:
		if input_sentence:
			input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
			if not input_length:
				input_length=100
			try:
				output = model.generate(input_ids, max_length=input_length)
			except:
				output = model.generate(input_ids, max_length=int(input_length))

			text = tokenizer.decode(output[0], skip_special_tokens=True)
			return html.Div([dcc.Markdown(text)])
	else:
  		return dash.no_update
		
if __name__ == '__main__':
	app.run_server(debug=True)
