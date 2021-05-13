from flask import Flask, render_template, request, jsonify
import tensorflow as tf
#from tensorflow import keras
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoder_model = tf.keras.models.load_model('./encoderModel')
decoder_model = tf.keras.models.load_model('./decoderModel')
with open('tokenizer_inputs.pickle', 'rb') as handle:
        tokenizer_inputs = pickle.load(handle)

word2idx_outputs = np.load('word2idx_outputs.npy',allow_pickle='TRUE').item()
idx2word_trans = np.load('idx2word_trans.npy',allow_pickle='TRUE').item()
LATENT_DIM_DECODER = 200
max_len_input = 8
max_len_target = 22

def decode_sequence(input_seq):
  # Encode the input as state vectors.
  enc_out = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))
  
  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']


  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, LATENT_DIM_DECODER))
  c = np.zeros((1, LATENT_DIM_DECODER))


  # Create the translation
  output_sentence = []
  for _ in range(max_len_target):
    o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        

    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)


#init_app
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('main.html')

@app.route('/chatbot', methods=['POST'])
def home():
    #inputStr = "What is our professsor email" # What is assignment 1 due date?
    inputStr = request.form['question']
    inputTexts = []
    inputTexts.append(inputStr)
    inputSequences = tokenizer_inputs.texts_to_sequences(inputTexts)
    encoderInputs = pad_sequences(inputSequences, maxlen=max_len_input)
    translation = decode_sequence(encoderInputs)
    print(encoderInputs)
    print(translation)
    return render_template('answer.html', answer=translation)

if __name__ == '__main__':
    app.run(debug=True)
