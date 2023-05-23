import os, glob, random, json, openai, tiktoken, pinecone
from datetime import datetime as dt, timedelta

start = dt.now()

# OPENAI text-embedding-ada-002 COST INFO
# $0.002 / 1000 tokens

# Define global openai variables
OPENAI_KEY = os.environ['OPENAI_KEY']
EMBEDDING_MODEL = 'text-embedding-ada-002'
CHAT_COMPLETION_MODEL = 'gpt-3.5-turbo'

# Define global pinecone variables
PINECONE_KEY = os.environ['PINECONE_KEY'] 
PINECONE_ENV = 'us-central1-gcp'
PINECONE_INDEX_NAME = 'baseindex'
PINECONE_NAMESPACE = 'wallstreetoddssupport'

# Define folder containing support snippets
SNIPPETS_FOLDER = 'support_snippets'

# Define user message max token size (user message includes the query + the support snippets)
USER_MESSAGE_MAX_TOKENS = 3584 # Max token size for GPT is 4096 (total between system message, query + support snippets and response), so this provides a buffer of 512 tokens for the system message and the response.

# Define the system message sent to CHAT_COMPLETION_MODEL on every query
SYSTEM_MESSAGE = 'You are a chatbot that helps with questions related to the website WallStreetOdds.com, which is a website that provides advanced trading data and tools for investors and day traders. Each question you are asked will come with potentially relevant information that may or may not help you answer the question. Use your best judgement in deciding if all, part or none of the provided information is relevant in answering the question. If you are fairly certain the question has nothing to do with WallStreetOdds.com, respond "Please ask only WallStreetOdds related questions or try rephrasing your question.". Otherwise, provide an answer to the question. If you are unable to find an exact answer or solution, please clarify to the user that you do not have an exact answer or solution and proceed to provide the best alternative answer or solution. If there is a moderate degree of uncertainty, let the user know you are unsure.'

# Connect to openai api
openai.api_key = OPENAI_KEY

# Connect to pinecone db and index (create index if it does not already exist)
pinecone.init(api_key=PINECONE_KEY,environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

# Create a function that helps us track how many tokens each support snippet will cost using tiktoken library
def num_tokens_from_string(string, encoding_name='cl100k_base'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Create a function that helps us find relavent support snippets based on the question
def find_similar(text,namespace,topK,values=False,metadata=False):
    """Returns vectors in a pinecone index that are similar to a given text."""
    # First embed the text (the question)
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    # Search pinecone for relavent support snippets (by searching for most similar embeddings using cosine similarity)
    query_response = index.query(
        namespace=namespace,
        top_k=topK,
        include_values=values,
        include_metadata=metadata,
        vector=embeddings
    )

    # Return relavent support snippets
    return query_response['matches']

# Creat the function that will actually be able to answer a question based on our cutom knowledge base (our support snippets)
def get_question_response(question):
    """Answers a question based on custom knowledge base (our support snippets)."""

    question_tokens = num_tokens_from_string(question)
    if question_tokens > USER_MESSAGE_MAX_TOKENS:
        print("Your question is of size ",question_tokens," tokens which exceeds the defined user message max tokens of ",USER_MESSAGE_MAX_TOKENS)
    else:
        # Get the relavent support snippets ids
        support_snippet_ids = [i['id'] for i in find_similar(question,PINECONE_NAMESPACE,5)]
        
        # Instatiate the user_message (question + support snippets) that will be passed along to chat completion model
        user_message = 'Question: '+question+' Here is potentially relavent content that may or may not help in answering the question. Please answer the question assuming the person asking the question does not know this information was shared with you.'

        # Add support snippets to the user message
        for support_snippet_id in support_snippet_ids:
            # Get support snippet file based on support snippet id
            support_snippet_file = json.loads(open(glob.glob(SNIPPETS_FOLDER+'/*_'+support_snippet_id+'.json')[0], "r").read())

            # Add the support snippet to the user message as context, if the support snippet won't result in the user message now exceeding the total allowed tokens
            support_snippet_tokens = support_snippet_file['tokens']
            if (question_tokens + support_snippet_tokens) < USER_MESSAGE_MAX_TOKENS:
                user_message += ' ' + support_snippet_file['content']
                question_tokens += support_snippet_tokens
            else:
                # Support snippet is too big to pass along
                pass
        
        # Send the question to chat completion model
        response = openai.ChatCompletion.create(
            model=CHAT_COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            temperature=0
        )

        # Print the cost of the chat completion model call for informative purposes
        tokens_used = response['usage']['total_tokens']
        print('Query took ',tokens_used,' tokens, which cost $',round((0.002 / 1000)*tokens_used,8))
        
        # Pass along the response content
        return response['choices'][0]['message']['content']

# Example response to a question
print(get_question_response("What are one day up odds?"))

print('script took: ', dt.now() - start)