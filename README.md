# University Chatbot

A Python-based conversational AI designed to handle university-related queries. This chatbot helps students and visitors find information about admissions, courses, faculty, events, and other campus inquiries efficiently.

## 🚀 Features

- **Intent Recognition**: Understands user queries related to various university topics.
- **Natural Language Processing**: Uses NLTK (Natural Language Toolkit) for tokenization and lemmatization.
- **Deep Learning Model**: (If applicable) Uses a Neural Network (TensorFlow/Keras) to classify intents.
- **User-Friendly Interface**: (If using Flask/GUI) Provides a web-based or graphical interface for interaction.
- **Easy Customization**: Responses and intents can be easily modified via a JSON file.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - `nltk` (Natural Language Processing)
  - `tensorflow` / `keras` (Neural Network Training)
  - `numpy` (Data manipulation)
  - `flask` (Web Framework - *if applicable*)
- **Data Format**: JSON (for storing intents and responses)

## 📂 Project Structure

```text
University_Chatbot/
│
├── intents.json          # Database of patterns, tags, and responses
├── chatbot_model.h5      # Trained model file (generated after training)
├── train.py              # Script to train the neural network model
├── app.py                # Main application script (GUI or Web)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── static/ & templates/  # (If using Flask) Web assets
