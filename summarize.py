import openai
import tiktoken
import numpy as np
import os
from summarize import sumsecrets

openai.api_key = sumsecrets.myAPIkey
encoding_break_status = False #Don't encode by default
path = os.getcwd()

message_history = [{"role": "assistant", "content": f"OK"}]

'''
Each of these lists contains, for each filetype, 3 prompts: 
1. Normal prompt for queries that are sent in full in one message
2. Beginning prompt for query split into multiple messages
3. Follow-up prompt for query split into multiple messages
'''
txtprompt = ["Please summarize this in a way that is useful to somebody looking back at a previous text file to remember what subjects were discussed, with a list of subjects, each subject entry supplemented with a quick summary of the discussion of each subject",
             "I have a conversation split into multiple parts. Please summarize each part in a way that is useful to somebody looking back at previous conversations to remember what subjects were discussed, by listing subjects covered and then supplementing each subject with a summary of the things said about each subject, in a NON-NUMBERED format (for example, using dashes instead of numbers to precede each subject)",
             "Here's the next part to summarize: "]
pyprompt = ["Please read this code and give me a detail explanation of, based on the code and comments, what this program is meant to do and how it does it.",
            "I have a program split into multiple parts. Please read this code and give me a detail explanation of, based on the code and comments, what this program is meant to do and how it does it."
            "Here's the rest of the code: "]

'''
The following function is used to process a file that takes up more tokens than is allowed,
Returns the rest of the file's content (rest_content) that hasn't been added to the matrix.
'''
def encoding_break(transcript, token_breaker):
    global encoding_break_status
    num = len(transcript) // token_breaker
    array = np.zeros((num,token_breaker), dtype=np.int64)
    remainder = len(transcript) % token_breaker
    print(remainder)
    if remainder > 0:
        rest_content = np.zeros((1, remainder), dtype=np.int64)
        rest_content[0] = transcript[num*token_breaker:]
        print(rest_content)
    for i in range(num):
        a = i*token_breaker
        c = (i+1)*token_breaker
        array[i:]=transcript[a:c]
        print(array)
    return array, rest_content

def open_file(path, filename, token_breaker):
    #Opens transcript file and loads encoding model
    transcript = open(f"{path}\\{filename}", "r").read()
    print('Transcript received!')
    encode = tiktoken.encoding_for_model('gpt-3.5-turbo')
    encode_list = encode.encode(transcript)
    print(f"The token number for the transcript is {len(encode_list)} tokens.")
    '''
    Check to see if the transcript is over our token limit, and if it is, slice it up to form a new numpy matrix
    if it is within the token limit, put it as it is into a matrix.
    '''
    if len(encode_list) > token_breaker:
        encoding_break_status = True
        final_list = encoding_break(encode_list, 3000)[0]
        remain_content = encoding_break(encode_list, 3000)[1]
        print("Separation process initiated...")
    else:
        encoding_break_status = False
        final_list = np.zeros((1,len(encode_list)), dtype=np.int64)
        final_list[0] = encode_list
        print("No separation needed!")
        remain_content = "none"
    return final_list, remain_content

'''
Starting to pass the arguments into the GPT model
This is modified from the original one--instead of summarizing each slice of the conversation,
which could result in missing context between slices if it's slic
'''


'''
For the following function, we pass our transcripts(sliced up) in the matrix into the gpt-3.5 models one by one
'''
def GPT(input, temperature, filetype, filename):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # tokenize the new input sentence
    if filetype == "txt":
        message_history.append({"role": "user", "content": f"{txtprompt[0]}: {input}"})
    elif filetype == "py":
        message_history.append({"role": "user", "content": f"{pyprompt[0]}: {filename}\n{input}"})
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=prompt_history,
      temperature=temperature
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

def GPTsplitfirst(input, temperature, filetype, filename):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # tell chatGPT that the split transcript will be sent in multiple parts and send the first part
    if filetype == "txt":
        message_history.append({"role": "user", "content": f"{txtprompt[1]}: {input}"})
    elif filetype == "py":
        message_history.append({"role": "user", "content": f"{pyprompt[1]}: {filename}\n{input}"})
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=prompt_history,
      temperature=temperature
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

def GPTsplitrest(input, temperature, filetype):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # send the rest of the parts of the split transcript
    if filetype == "txt":
        message_history.append({"role": "user", "content": f"{txtprompt[2]}: {input}"})
    elif filetype == "py":
        message_history.append({"role": "user", "content": f"{pyprompt[2]}: {input}"})
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=prompt_history,
      temperature=temperature
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

'''
We can then sum up all the processed summarization to a str(final_sum)
'''
def summarize(path, filename, temperature, filetype):
    final_list, remain_content = open_file(path, filename, 3000)
    encode = tiktoken.encoding_for_model('gpt-3.5-turbo')
    final_sum = 'Summary\n\n'
    if encoding_break_status is True: #Check to see if we used the encoding_break function, if true, we process transcripts one by one. Otherwise, we just feed the original transcript.
        for i in range(len(final_list)):
            if i == 0:
                print(f'Processing paragraph {i + 1} with temperature {temperature}...')
                final_sum += "Part 1:\n"
                final_sum += GPTsplitfirst(encode.decode(final_list[i]), temperature, filetype, filename)
            else:
                print(f'Processing paragraph {i + 1} with temperature {temperature}...')
                final_sum += f"\n\nPart {i + 1}:\n"
                final_sum += GPTsplitrest(encode.decode(final_list[i]), temperature, filetype)
        print(f'Processing the last paragraph with temperature {temperature}...')
        final_sum += f"\n\nFinal part:\n"
        final_sum += GPTsplitrest(encode.decode(remain_content[0]), temperature, filetype)
    else:
        print(f'Processing file with temperature {temperature}...')
        final_sum +=GPT(encode.decode(final_list[0]), temperature, filetype, filename)
    if filetype == "txt":
        rawfilename = filename.removesuffix(".txt")
    elif filetype == "py":
        rawfilename = filename.removesuffix(".py")
    with open(f"{path}\\{rawfilename}_summary.txt", "w", encoding="utf-8-sig") as file:
            text = file.write(final_sum)
            file.close()

