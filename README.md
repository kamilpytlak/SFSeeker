
<h1 align="center">
  <br>
  <img src="img/logo.jpg" alt="SFSeeker logo" width="250">
  <br>
  Sci-Fi Seeker
  <br>
</h1>

<h4 align="center">An AI assistant with a semantic engine/question writing tutor in Sci-Fi Stack Exchange service<br>built on top of <a href="https://streamlit.io/" target="_blank">Streamlit</a>.</h4>

<p align="center">
  <a href="#key-features">Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#contact">Contact</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
SF Seeker is an AI assistant designed for Sci-Fi Stack Exchange, utilizing an all-MiniLM-L6-v2 language model. It helps users improve their question-writing skills and find similar questions on the Sci-Fi Stack Exchange website. This tool leverages a database of 71,013 questions to locate semantically similar questions, reducing the likelihood of creating duplicate threads. Additionally, SF Seeker is in the process of developing a feature that identifies words in questions that affect the likelihood of receiving answers, assisting users in formulating more precise inquiries. This feature uses a model trained with gradient reinforcement based on TF-IDF features.
</p>

## Features

*  🔎 Based on a database of 71,013 questions, it searches for the most semantically similar questions to the one entered by the user. This supports the process of fiding the same/similar questions already asked and prevents the creation of duplicate threads.
*  👨‍⚕️ [IN PROGRESS] Indicates words in a question that have a negative and positive effect on the chance of getting an answer. It supports the process of arranging more precise questions. A model based on gradient reinforcement learned using TF-IDF features was used.

## How To Use

Currently, there is a way to use this app locally by cloning the repository (using git or by downloading it directly from the website), install the dependencies from the configuration file `Pipfile` and launch the app locally using a browser.

```bash
# Clone this repository
$ git clone https://github.com/kamilpytlak/SFSeeker

# Go into the repository
$ cd SFSeeker

# Install pipenv (in case it's not installed) and, run pipenv shell and install dependencies
$ pip install pipenv
$ pipenv shell
$ pipenv install

# Ensure that the streamlit package was installed successfully.
$ streamlit hello

# Finally, run the app locally
$ streamlit run ./main.py
```

## Contact

If you have any problems, ideas or general feedback, please don't hesitate to contact me at [kam.pytlak@gmail.com](mailto:kam.pytlak@gmail.com). I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/#)
- [scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/)
- [xgboost](https://xgboost.readthedocs.io/en/latest/index.html)
- [sentence-transformers](https://www.sbert.net/)
- 
## License
MIT

---

> GitHub [@kamilpytlak](https://github.com/kamilpytlak) &nbsp;&middot;&nbsp;
> LinkedIn [kamil-pytlak](https://www.linkedin.com/in/kamil-pytlak/)

