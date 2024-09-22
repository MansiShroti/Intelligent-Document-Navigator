import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt') 
nltk.download("stopwords")
import PyPDF2
import os
import time
import string
string.punctuation
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv() 
API_KEY=os.getenv("API_KEY")
MODE = os.getenv("MODE")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")


pc = Pinecone(api_key=API_KEY)
# UPLOAD_DIR = Path() / 'uploads'
# UPLOAD_DIR = '/opt/source-code/uploads'

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'semantic-search'
stopwords = nltk.corpus.stopwords.words('english')


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output


async def uploader(file,model):
    content = ""
    filename = file.filename
    print(file.content_type)
    if file.content_type=="application/pdf":
        data = await file.read()
        print(filename)
        save_to = UPLOAD_DIR + '/' + filename
        with open(save_to, 'wb') as f:
            f.write(data)
        
        pdf_reader = PyPDF2.PdfReader(save_to)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            content += page.extract_text()

        os.remove(save_to)
    elif file.content_type=="text/plain":
        content = await file.read()
        content=content.decode('utf-8')
    else:
        return f"Send {filename} file in either PDF format or text format".format(filename=filename)
    
    sent_tkn = nltk.sent_tokenize(content)
    sent_tkn = list(map(lambda x:remove_punctuation(x),sent_tkn))
    sent_tkn = list(map(lambda x: x.lower(),sent_tkn))

    sent_tkn = list(map(lambda x:x.replace("\\",""),sent_tkn))
    sent_tkn = list(map(lambda x:x.replace("'",""),sent_tkn))
    sent_tkn = list(map(lambda x:x.replace("\n",""),sent_tkn))
    sent_tkn = list(map(lambda x:' '.join(x.split()),sent_tkn))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=model.get_sentence_embedding_dimension(),
            metric='cosine',
            spec=spec
        )
        
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    #connect to index
    index = pc.Index(index_name)

    batch_size = 128

    namespaces = [i for i in index.describe_index_stats()['namespaces']]

    if MODE!="test":
        if filename in namespaces:
            return f"A file by '{filename}' name already exists".format(filename=filename)

    for i in tqdm(range(0, len(sent_tkn), batch_size)):
        # find end of batch
        i_end = min(i+batch_size, len(sent_tkn))
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'text': text} for text in sent_tkn[i:i_end]]
        # create embeddings
        xc = model.encode(sent_tkn[i:i_end])
        # create records list for upsert
        records = zip(ids, xc, metadatas)
        # upsert to Pinecone
        if MODE !="test":
            index.upsert(vectors=records,namespace = filename)
    
    return 'f{filename} successfully upserted'.format(filename=filename)

    
async def query(query,model,threshold):
    xq = model.encode(query).tolist()
    # print(xq)
    ans=[]

    #connect to index
    index = pc.Index(index_name)
    namespaces = [i for i in index.describe_index_stats()['namespaces']]
    for namespace in namespaces:
        xc = index.query(vector=xq,top_k=1, include_metadata=True,namespace=namespace)
        # print(xc)
        sum=0
        k=0
        print(xc)
        print()
        print()
        i = xc['matches']
        for j in i:
            sum+=j['score']
            k+=1

        avg = sum/k

        print(threshold)

        if(avg >= threshold):
            ans.append(namespace)

    return ans

    

