import os, glob, random, json, openai, tiktoken, pinecone
from datetime import datetime as dt
import pandas as pd

# OPENAI text-embedding-ada-002 COST INFO
# $0.0004 / 1000 tokens = ~3000 pages per dollar

start = dt.now()

# Define global openai variables
OPENAI_KEY = os.environ['OPENAI_KEY']
EMBEDDING_MODEL = 'text-embedding-ada-002'

# Define global pinecone variables
PINECONE_KEY = os.environ['PINECONE_KEY'] 
PINECONE_ENV = 'us-central1-gcp'
PINECONE_INDEX_NAME = 'baseindex'
PINECONE_INDEX_EMBEDDING_DIMS = 1536 # corresponds to text-embedding-ada-002 embedding dims
PINECONE_INDEX_METRIC = 'cosine' # corresponds with similarity search metric that works best with text-embedding-ada-002 embeddings
PINECONE_NAMESPACE = 'wallstreetoddssupport'

# Define folder containing support snippets
SNIPPETS_FOLDER = 'support_snippets'

# Define name/path of file that will contain a summary of our summary snippets
SUMMARY_SNIPPETS_FILE_NAME = SNIPPETS_FOLDER + '/snippets_summary.csv'

# Define max support snippet token size
SUPPORT_SNIPPET_MAX_TOKENS = 3072 # Max token size for GPT is 4096 (total between system message, query + support snippets and response), so this provides a buffer of 1024 tokens for the system message, the query and the response.

# Connect to openai api
openai.api_key = OPENAI_KEY

# Connect to pinecone db and index (create index if it does not already exist)
pinecone.init(api_key=PINECONE_KEY,environment=PINECONE_ENV)
if not PINECONE_INDEX_NAME in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=PINECONE_INDEX_EMBEDDING_DIMS, metric=PINECONE_INDEX_METRIC)
index = pinecone.Index(PINECONE_INDEX_NAME)

# Create a function that helps us track how many tokens each support snippet will cost using tiktoken library
def num_tokens_from_string(string, encoding_name='cl100k_base'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Create a function that helps us create a unique id for each of our support snippets
def create_id(digits):
    """Returns a case-sensitive alphanumeric randomly generated string."""
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(characters) for i in range(digits))

# Instantiate a list that will hold summary data for all of our support snippets (will pull existing summary data for support snippets already embedded if any exist)
if os.path.exists(SUMMARY_SNIPPETS_FILE_NAME):
    files_summary_data = pd.read_csv(SUMMARY_SNIPPETS_FILE_NAME).values.tolist()
else:
    files_summary_data = []

# Get the name of all the support snippet files to be embedded
files_to_embed = glob.glob(SNIPPETS_FOLDER+'/*.txt')

# Iterate through each support snippet file, embed the file's text content in pinecone, and convert the file into a json file containing the content's token size and the content itself
for file_to_embed in files_to_embed:

    # Open snippet file and capture content
    with open(file_to_embed, "r") as file:
        content = str(file.read())

    # Get the token size of the support snippet
    tokens = num_tokens_from_string(content)

    # Check if support snippet satisfies max tokens requirement (alert user if it doesn't, proceed if it does)
    if tokens > SUPPORT_SNIPPET_MAX_TOKENS:
        print(file_to_embed," is of size ",tokens," tokens which exceeds the support snippet defined max tokens of ",SUPPORT_SNIPPET_MAX_TOKENS)
    else:
        # Get file name which serves as the title of the support snippet
        title = file_to_embed.split('\\')[1].split('.')[0]

        # Create a unique id for the support snippet
        unique_id = create_id(8)

        # Embed the snippet using openai
        embedded_snippet = openai.Embedding.create(input=content,model=EMBEDDING_MODEL)['data'][0]['embedding']

        # Add the embedded snippet into pinecone db index
        index.upsert(vectors=[{'id': unique_id, "values": embedded_snippet}],namespace=PINECONE_NAMESPACE)

        # Create json for snippet
        json_file_content = {"tokens": tokens, "content": content}

        # Save json_file_content as a json file and remove txt file
        file_name = title+'_'+unique_id+'.json'
        with open(SNIPPETS_FOLDER+"/"+file_name, "w") as file:
            json.dump(json_file_content,file)
        os.remove(file_to_embed)

        # add support snippet summary data to files_summary_data
        files_summary_data.append([unique_id,tokens,title])

# Save support snippets summary data in a csv file
df = pd.DataFrame(data=files_summary_data,columns=['unique_id','tokens','title'])
df.to_csv(SUMMARY_SNIPPETS_FILE_NAME,index=False)

print('script took: ', dt.now() - start)