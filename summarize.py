import openai
import tiktoken
import numpy as np
import os
from summarize import sumsecrets

openai.api_key = sumsecrets.myAPIkey
encoding_break_status = False #Don't encode by default
path = os.getcwd()

message_history = [{"role": "assistant", "content": f"OK"}]
prompt1 = "Please summarize this in a way that is useful to somebody looking back at previous conversations to remember what subjects were discussed, with a list of subjects, each subject entry supplemented with a quick summary of the discussion of each subject"
prompt2 = "I have a conversation split into multiple parts. Please summarize each part in a way that is useful to somebody looking back at previous conversations to remember what subjects were discussed, by listing subjects covered and then supplementing each subject with a summary of the things said about each subject, in a NON-NUMBERED format (for example, using dashes instead of numbers to precede each subject)"

'''
The following function is used to process a file that takes up more tokens than is allowed,
Returns the rest of the file's content (rest_content) that hasn't been added to the matrix.
'''
def encoding_break(transcript, token_breaker):
    global encoding_break_status
    num = len(transcript) // token_breaker
    array = np.zeros((num,token_breaker), dtype=np.int64)
    remainder = len(transcript) % token_breaker
    if remainder > 0:
        rest_content = np.zeros((1, remainder), dtype=np.int64)
        rest_content[0] = transcript[num*token_breaker:]
    for i in range(num):
        a = i*token_breaker
        c = (i+1)*token_breaker
        array[i:]=transcript[a:c]
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
def GPT(input):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"{prompt1}: {input}"}) # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

def GPTsplitfirst(input):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # tell chatGPT that the split transcript will be sent in multiple parts and send the first part
    message_history.append({"role": "user", "content": f"{prompt2}: {input}"}) # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

def GPTsplitrest(input):
    message_history = [{"role": "assistant", "content": f"OK"}]
    # send the rest of the parts of the split transcript
    message_history.append({"role": "user", "content": f"Next part to summarize: {input}"}) # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

'''
We can then sum up all the processed summarization to a str(final_sum)
'''
def summarize(path, filename):
    final_list, remain_content = open_file(path, filename, 3000)
    encode = tiktoken.encoding_for_model('gpt-3.5-turbo')
    final_sum = 'Summary\n\n'
    if encoding_break_status is True: #Check to see if we used the encoding_break function, if true, we process transcripts one by one. Otherwise, we just feed the original transcript.
        for i in range(len(final_list)):
            if i == 0:
                print(f'Processing paragraph {i + 1}...')
                final_sum += "Part 1:\n"
                final_sum += GPTsplitfirst(encode.decode(final_list[i]))
            else:
                print(f'Processing paragraph {i + 1}...')
                final_sum += f"\n\nPart {i + 1}:\n"
                final_sum += GPTsplitrest(encode.decode(final_list[i]))
        print(f'Processing the last paragraph...')
        final_sum += f"\n\nFinal part:\n"
        final_sum += GPTsplitrest(encode.decode(remain_content[0]))
    else:
        print(f'Processing the transcript...')
        final_sum +=GPT(encode.decode(final_list[0]))

    rawfilename = filename.strip(".txt")
    with open(f"{path}\\{rawfilename}_summary.txt", "w", encoding="utf-8-sig") as file:
            text = file.write(final_sum)
            file.close()

