This AI HealthCare Chatbot is my 2nd Year project in which it understands the intends of the user and give the solution according to it. 
    
    First Install the required the libraries 

  pip install tensorflow 
  pip install keras 
  pip install pickle
  pip install nltk
  pip install flask

  Now we are going to build the chatbot using Flask framework but first, let us see the file structure and the type of files we will be creating:

data.json – The data file which has predefined patterns and responses.
trainning.py – In this Python file, we wrote a script to build the model and train our chatbot.
Texts.pkl – This is a pickle file in which we store the words Python object using Nltk that contains a list of our vocabulary.
Labels.pkl – The classes pickle file contains the list of categories(Labels).
model.h5 – This is the trained model that contains information about the model and has weights of the neurons.
app.py – This is the flask Python script in which we implemented web-based GUI for our chatbot. Users can easily interact with the bot.

    In this chatbot you can upload any photo , so you can feel that you are talking to your guy
