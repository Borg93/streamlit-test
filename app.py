import pandas as pd
import streamlit as st
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from samples import texts
import urllib.request




st.set_page_config(page_title='Word Extractor',page_icon=":heart:")

st.title("Atkins Keyword extractor")

add_selectbox = st.sidebar.selectbox(
    "Choice model:",
    ("paraphrase-multilingual-mpnet-base-v2", "Gabriel/Model_Atkins", "Contrastive-Tension/BERT-Base-Swe-CT-STSb")
)

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)
def load_model():

    if add_selectbox == "Gabriel/Model_Atkins":
           BERT = TransformerDocumentEmbeddings(add_selectbox)
           model = KeyBERT(model=BERT)
    
    elif add_selectbox == "Contrastive-Tension/BERT-Base-Swe-CT-STSb":
           BERT = TransformerDocumentEmbeddings(add_selectbox)
           model = KeyBERT(model=BERT)

    else:
        model = KeyBERT(add_selectbox)

    return model

check = st.checkbox('Run model', value=True)
st.write('State of model:', check)
if check:
    model = load_model()
    st.write('Model used:', add_selectbox )




placeholder = st.empty()
text_input = placeholder.text_area("Type in some text you want to analyze", height=300)

sample_text = st.selectbox(
    "Or pick some sample texts", [f"sample {i+1}" for i in range(len(texts))]
)

sample_id = int(sample_text.split(" ")[-1])
text_input = placeholder.text_area(
    "Type in some text you want to analyze", value=texts[sample_id - 1], height=400
)


top_n = st.sidebar.slider("Select number of keywords to extract", 5, 20, 10, 1)
min_ngram = st.sidebar.number_input("Min ngram", 1, 5, 1, 1)
max_ngram = st.sidebar.number_input("Max ngram", min_ngram, 5, 3, step=1)
st.sidebar.code(f"ngram_range = ({min_ngram}, {max_ngram})")



params = {
    "docs": text_input,
    "top_n": top_n,
    "keyphrase_ngram_range": (min_ngram, max_ngram),
    "stop_words": "english",
}

add_diversity = st.sidebar.checkbox("Add diversity to the results")

if add_diversity:
    method = st.sidebar.selectbox(
        "Select a method", ("Max Sum Similarity", "Maximal Marginal Relevance")
    )
    if method == "Max Sum Similarity":
        nr_candidates = st.sidebar.slider("nr_candidates", 20, 50, 20, 2)
        params["use_maxsum"] = True
        params["nr_candidates"] = nr_candidates

    elif method == "Maximal Marginal Relevance":
        diversity = st.sidebar.slider("diversity", 0.1, 1.0, 0.6, 0.01)
        params["use_mmr"] = True
        params["diversity"] = diversity


keywords = model.extract_keywords(**params)

def remove_stop_words(sentence):
    url = "https://gist.githubusercontent.com/peterdalle/8865eb918a824a475b7ac5561f2f88e9/raw/cc1d05616e489576c1b934289711f041ff9b2281/swedish-stopwords.txt"
    file = urllib.request.urlopen(url)

    stopword =[]
    for line in file:
        decoded_line = line.decode("utf-8")
        stopword.append(decoded_line.strip('\n'))
    
    word_list=sentence.split()
    clean_sentence=' '.join([w for w in word_list if w.lower() not in stopword])
    return(clean_sentence)


if add_selectbox == "Gabriel/Model_Atkins" or add_selectbox == "Contrastive-Tension/BERT-Base-Swe-CT-STSb":
    if keywords != []:

        st.info("Extracted Swedish keywords")
        keywords = pd.DataFrame(keywords, columns=["keyword", "relevance"])
        keywords['Without stopwords']=keywords['keyword'].apply(remove_stop_words)
        
        #keywords['keyword'].apply(lambda x: [item for item in x if item not in stopwords])
        st.table(keywords)
    
else:
    if keywords != []:
        st.info("Extracted keywords (multilingual model)")
        keywords = pd.DataFrame(keywords, columns=["keyword", "relevance"])
        st.table(keywords)

