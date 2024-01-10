
# Bangla Hate Speech Detection 

This is a project to detect Bangla Hate Speech. It is also intrigated with a demo website using FAST API.

In this process **Deep Learning model** (Language Model) **LSTM** is used.

## NOTE
Please at first use the **Hate_LSTM.ipynb** file to get the necessary h5 and pickle files. Then add these files in the root directory. 

I could not upload these files due to the limitation of free storage.

## Run Locally

1. Clone the project

```bash
  git clone https://github.com/sabrina-mumu/Bengali-Hate-Speech-Detection
```

2. Go to the project directory

```bash
  cd my-project
```
Here, replace my-project with the path of your project

3. Create Virtual Environment
```bash
  pip install virtualenv
  python -m venv venv
  .\venv\Scripts\activate
```

4. Install necessary libraries
```bash
  pip install flask
  pip install tensorflow
```

5. Start the server
```bash
  python api_connect.py
```

##Output

![Hate Comment Detection](https://github.com/sabrina-mumu/Bengali-Hate-Speech-Detection/blob/master/hatecomment.png?raw=true)

![Non Hate Comment Detection](https://github.com/sabrina-mumu/Bengali-Hate-Speech-Detection/blob/master/nonhatecomment.png?raw=true)
