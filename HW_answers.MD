---
title: "HW3: Calling an LLM with Bedrock"
author: "Jaren Figueiredo"
date: "2026-04-18"
---

Github repo can be found [here](https://github.com/jfigge7/gsb-570-gen-ai-hw-3/tree/master)

## 1. What happens when you ask the model what the current day is or when were you trained? Is this expected behaviour?
(Using `deepseek.v3.2`)
Current day:
```
Query: What is the current day?
Model Answer: I am unable to determine the current day on my own. For the most accurate and up-to-date date, I recommend checking your device's clock, a calendar, or a search engine.
```

Training date:
```
Query: When were you trained?
Response: I don't have a specific "training date" in the traditional sense! I'm based on DeepSeek's latest model, which was developed by DeepSeek (深度求索). My knowledge cutoff date is July 2024.

However, it's important to note that I don't have real-time knowledge - I can't access current events or information after my training period. What I can do is draw upon the extensive knowledge I was trained on up to July 2024, and I also have a 128K context window which allows me to work with quite lengthy conversations and documents.   

If you're interested in more specific technical details about my training or architecture, I'd recommend checking DeepSeek's official documentation and announcements - they would have the most accurate and detailed information about how I was developed!  

Is there something specific about my capabilities or knowledge base you're curious about? 😊
```

I suppose this behavior is unexpected, because we expect the AI model to know everything (also because I assumed it could do a web-search for the date).  But then again, you've said in class that LLMs don't know the current date, so in that sense the model behavior is to be expected.

## 2. Try at least 3 other models besides Anthropic models in your code comment what you needed to change to call other models (use the command to find alternate modelIDs aws bedrock list-foundation-models --region us-west-2 --profile)
I used `aws bedrock list-foundation-models --region us-west-2 --profile GSB570-BedrockOnly-490332585640 --query "modelSummaries[?contains(providerName, '')].{Id:modelId, Name:modelName}" --output table` to list a table of usable models through BedRock.  Then, I copied over the Id value for a model I want to use into the `modelId` parameter for the `boto3.session().client().invoke_model()` method.
The models I used were
```
nvidia.nemotron-nano-12b-v2
qwen.qwen3-235b-a22b-2507-v1:0
openai.gpt-oss-120b-1:0
```
Nvidia model:
```
Query: When were you trained?; Model: nvidia.nemotron-nano-12b-v2
Response: My training data cutoff time is October 2024. If you have any questions or need assistance, feel free to let me know anytime!
```
QWEN model:
```
Query: When were you trained?; Model: qwen.qwen3-235b-a22b-2507-v1:0
Response: I was trained on a vast amount of text data from the internet up until 2024. This means my training data cutoff point is 2024, and I do not have information or training on events that occurred after that year. If you have any other questions, feel free to ask!
```

OpenAI model:
```
Query: When were you trained?; Model: openai.gpt-oss-120b-1:0
Response: <reasoning>The user asks: "When were you trained?" They want to know when the model was trained. Need to respond with info about training cutoff date and timeline. According to system message: "knowledge cutoff: 2024-06". So we can say training data up to June 2024, model released later (2024?). Mention that we are GPT-4. Provide context. Also ask if they want more details.</reasoning>I’m a GPT‑4 model that was trained on a broad mix of publicly available text, licensed data, and other sources up through **June 2024**. After that point, I haven’t been updated with new information (although I can still help you reason about events or trends that have occurred since then).

If you’d like more details about the training process, the kinds of data that were included, or how the cutoff date might affect the answers I can give, just let me know!
```

All of the modelIds were passed in successfully as arguments without needing to further modifythe code.  However, the OpenAI model seems to have displayed its reasoning, which I'm not sure if it's supposed to do.

## 3. Build a multi-turn conversation using the messages array.
I altered the `main()` method to the following:
```python
def main():
  session = boto3.Session(profile_name="GSB570-BedrockOnly-490332585640")
  identity = session.client("sts").get_caller_identity()
  print("Account:", identity["Account"])
  print("UserId:", identity["UserId"])
  print("Arn:", identity["Arn"])

  message_history = ""
  while True:
    query = None
    while query is None or query.strip() == "":
      query = input("\nEnter your query: ").strip()
    model = "openai.gpt-oss-120b-1:0"
    message_response = call_bedrock_model(message_history + f"\n\nHuman: {query}\n\nAssistant:", model) 
    print(f"\nQuery: {query}; Model: {model}")
    print(f"Response: {message_response}")
    message_history += f"\n\nHuman: {query}\n\nAssistant: {message_response}"
```
I wrapped the function call into a while True loop and added "context" to the message that the model receives as context.

As you can see, the model was able to look back at my previous query and tell me what I had asked:
```md
Enter your query: Teach me about transformers in AI, starting with the basics.

Query: Teach me about transformers in AI, starting with the basics.; Model: openai.gpt-oss-120b-1:0
Response: <reasoning>We need to respond as ChatGPT teaching about transformers in AI, starting with basics. Should be thorough, clear, maybe sections: intro, history, self-attention, architecture, training, variations, applications, resources. Should be engaging. Use simple language, analogies. Should ask follow-up? Probably just give a lesson.

We should follow the style guidelines. No disallowed content. Provide explanation. Could ask if they'd like deeper dive. But answer comprehensively.

Let's produce a structured answer.</reasoning>## Transformers — the “big picture” of modern AI

Below is a step‑by‑step guide that takes you from the very first ideas to the state‑of‑the‑art models you see in chat‑bots, translation services, image generators, and more. Feel free to stop at any point and ask follow‑up questions—there’s plenty of depth to explore!

---

## 1️⃣ Why we needed a new architecture

| **Task** | **Traditional approach** | **Problem** |
|----------|--------------------------|-------------|
| **Sequence modeling** (e.g., language, DNA) | Recurrent Neural Networks (RNNs), LSTMs, GRUs | Must process tokens one‑by‑one → slow, hard to capture long‑range dependencies, gradient vanishing/exploding. |
| **Parallel hardware** | RNNs are inherently sequential. | Modern GPUs/TPUs work best when many operations run *simultaneously* (parallelism). |
| **Context window** | Fixed‑size hidden state in RNNs → limited memory. | Long‑range relationships (e.g., “the *cat* that chased the *mouse* …”) get lost. |

**Transformers** (proposed in the 2017 paper *“Attention Is All You Need”* by Vaswani et al.) were designed to **eliminate recurrence** and **rely entirely on attention**, allowing:

- **Full parallelism** across all tokens.
- **Direct connections** between any two positions (no matter how far apart).
- **Scalable training** on massive data sets.

---

## 2️⃣ Core building block: **Self‑Attention**

### 2.1 Intuition
Imagine you’re reading a sentence and you want to understand the word “bank”. You naturally look at surrounding words: “river **bank**” versus “money **bank**”.

Enter your query: Please relay back to me my previous query.

Query: Please relay back to me my previous query.; Model: openai.gpt-oss-120b-1:0
Response: <reasoning>The user asks: "Human: Please relay back to me my previous query.

Assistant:" They want the assistant to relay back their previous query. The previous query from the human is the request: "Teach me about transformers in AI, starting with the basics." So we should repeat that. Also maybe confirm. The instruction: "Human: Please relay back to me my previous query." So we should output the previous query. We'll output exactly the query.</reasoning>Your previous query was:

**“Teach me about transformers in AI, starting with the basics.”**
```

Of course, the first response cuts off its answer when it hits the `max_tokens` limit, currently set at 500 tokens.

## 4. Use a classifier to decide which LLM you will call for an answer. For complex questions choose a simple model and for complex queries call a more complex model. What else do you think you could use a classifier for?
I wrote a classifier (actually not using AI believe it or not), and the code can be found in [my github repo](https://github.com/jfigge7/gsb-570-gen-ai-hw-3/tree/master), which used [classify_prompt.py](https://github.com/jfigge7/gsb-570-gen-ai-hw-3/blob/master/classify_prompt.py) to classify prompts and [call_model_hw2.py](https://github.com/jfigge7/gsb-570-gen-ai-hw-3/blob/master/call_model_hw2.py) to actually call the model based on the user query. The system prompt can be found in [classfier_instructions.txt](https://github.com/jfigge7/gsb-570-gen-ai-hw-3/blob/master/classifier_instructions.txt), which I had AI write for me.  I could write a classfier for a ton of prompts.  For instance, if I want a homework helper, my classifier could determine the type of problem the user wants 
help with, and based on the classifier, I could alter the system prompt to steer the LLM towards the subject in question.
Examples:
```
Enter your query: How many quarts are in a gallon
Classified message complexity as: simple

Query: How many quarts are in a gallon; Model: nvidia.nemotron-nano-12b-v2
Response: 1 gallon equals 4 quarts. This is a standard conversion in the U.S. liquid measurement system.
```
```
Enter your query: How do I get a job?
Classified message complexity as: complex

Query: How do I get a job?; Model: qwen.qwen3-235b-a22b-2507-v1:0
Response: Getting a job involves several steps, and the exact path depends on your skills, experience, industry, and goals. Here’s a step-by-step guide to help you land a job:

### 1. **Self-Assessment**
   - **Identify your skills and strengths**: What are you good at? What do you enjoy doing?
   - **Define your career goals**: What kind of job are you looking for? Full-time, part-time, remote, in-person?
   - **Determine your field/industry**: Research industries that match your background or interests.

### 2. **Update Your Resume**
   - Tailor your resume for the types of jobs you want.
   - Highlight relevant experience, skills, education, and achievements.
   - Use clear, professional formatting and action verbs.
   - Keep it concise (usually 1–2 pages).
...
```