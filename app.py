from flask import Flask, render_template, request
import numpy as np
import torch
import torch.nn as nn
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
filename = "berdata.txt"
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
model = CharModel()
model.load_state_dict(best_model)


app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('home.html')

@app.route('/output', methods=['POST', 'GET'])
def output():
    """text = ""
    if request.method == 'POST':
        text = request.form['text']
    #return render_template('output.html')
    #start = np.random.randint(0, len(raw_text)-seq_length)
    #
    #prompt = filter(text.lower(), letters)
    prompt = text"""

    seq_length = 100
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    start = np.random.randint(0, len(raw_text)-seq_length)
    prompt = raw_text[start:start+seq_length]
    pattern = [char_to_int[c] for c in list(prompt)]

    model.eval()
    print('Prompt: "%s"' % prompt)
    output = ""
    with torch.no_grad():
        for i in range(256):
            # format input array of int into PyTorch tensor
            x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
            x = torch.tensor(x, dtype=torch.float32)
            # generate logits as output from the model
            prediction = model(x)
            # convert logits into one character
            index = int(prediction.argmax())
            result = int_to_char[index]
            output += result
            print(result, end="")
            # append the new character into the prompt for the next iteration
            pattern.append(index)
            pattern = pattern[1:]
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>BermanGPT 7.0</title>
  <link rel="stylesheet"  type="text/css" href="static/output.css">
  <h1 id="TitleH1"> BermanGPT 7.0 </h1>
</head>
<body>
  <div id="InputDiv"> 
    <img id="Berman1" src="static/berman1.jpg" alt="Image of Berman playing guitar"> <br>
    <!--<input id="BerInput" type="InputText" name="InputText"> <br>
    <button id="BerButton">Generate the text!</button> <br>
    -->

    <form action="/output">
      <input id="BerButton" type="submit" value="Generate a Bermanic text!">
    </form> 



  </div>

  <div id="OutputDiv">
    <img id="Berman2" src="static/berman2.jpg" alt="Insert image of berman here"> : {output}<br>



</body>"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
