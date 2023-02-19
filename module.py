from transformers import RagTokenizer 
from transformers import RagRetriever
from transformers import RagTokenForGeneration
from transformers import GPT2Tokenizer, GPTJForCausalLM,OPTForCausalLM
from datasets import load_dataset, load_from_disk
import torch
from rouge import Rouge 
import csv 
import logging
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import queue
from transformers import T5Tokenizer, T5ForConditionalGeneration
from model_utils import *

import ctranslate2


'''
    Model Classes
'''

# DPR model class
class rag_retreiver():
    def __init__(self,dataset_path, index_path,device):
        super(rag_retreiver,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="custom",
            passages_path=dataset_path,
            index_path=index_path,
        )
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True).to(self.device)

    def retreive(self,input_text:str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        # with tokenizer.as_target_tokenizer():
        #     targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # 1. Encode
        question_hidden_states = self.model.question_encoder(input_ids)[0]
        # 2. Retrieve
        docs_dict = self.retriever(input_ids.cpu().numpy(), question_hidden_states.cpu().detach().numpy(), return_tensors="pt")
        # docs_dict = self.retriever(input_ids, question_hidden_states, return_tensors="pt")
        doc_scores = torch.bmm(
            question_hidden_states.cpu().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1)
        return docs_dict, doc_scores
    
    def retreive_psg(self,input_text:str):
        docs_dict, doc_scores = self.retreive(input_text)
        doc_ids = docs_dict['doc_ids'][0].tolist()
        num_docs = len(doc_ids)
        total_doc = []
        for idx in doc_ids:
            total_doc.append(self.dataset[idx]['text'])
        return total_doc,docs_dict,doc_scores 

# OPT model class 
class opt_model():
    def __init__(self,model_path,device = torch.device("cuda:1"),ct2_path=None,is_server = False,device_index = [0,1,3],n_stream = 3):
        super(opt_model,self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if(ct2_path != None):
            if(is_server):
                self.model = ctranslate2.Generator(ct2_path, compute_type="float16",device = "cuda",device_index = device_index,inter_threads=n_stream,max_queued_batches = -1 )
                # self.model = ctranslate2.Generator(ct2_path, compute_type="float16",device = "cuda",device_index = [0],inter_threads=3, max_queued_batches = -1)
                # self.model = ctranslate2.Generator(ct2_path, compute_type="float16",device = "cuda",device_index = [0],inter_threads=3)
                self.deploy = "ct2-server"
            else:  
                self.model = ctranslate2.Generator(ct2_path, compute_type="float16",device = "cuda",device_index = self.device.index)
                self.deploy = "ct2"
        else:
            self.model = OPTForCausalLM.from_pretrained(model_path).to(self.device)
            self.deploy = "cuda"
        print(f"[OPT_MODEL] Deployment: {self.deploy}")
        if(self.deploy == "ct2-server"):
            print(f"[OPT_MODEL] Device Index: {device_index}")
            print(f"[OPT_MODEL] CUDA Streams: {n_stream}")
        
        
    def text_gen(self,input_text:str,max_len:int = 200) -> str: 
        if(self.deploy == "cuda"):
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            # outputs = self.model.generate(**inputs,max_length = max_len,do_sample = True,early_stopping = False,temperature=0.8, top_p = 0.9)
            outputs = self.model.generate(**inputs,penalty_alpha=0.6, top_k = 4,max_length=max_len)
            out_text = self.tokenizer.batch_decode(outputs,skip_special_tokens = True)[0]
            return out_text
        elif(self.deploy == "ct2"):
            start_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input_text))
            # results = self.model.generate_batch([start_tokens],beam_size=5,max_length = max_len)
            results = self.model.generate_batch([start_tokens], max_length=max_len, sampling_topk=10)
            output = self.tokenizer.decode(results[0].sequences_ids[0],skip_special_tokens = True)
            return output.replace("<s>","")
        else:
            start_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input_text))
            results = self.model.generate_batch([start_tokens], max_length=max_len, sampling_topk=10,asynchronous=True)
            while(results[0].done()!=True):
                pass
            return  self.tokenizer.decode(results[0].result().sequences_ids[0])
    def prepare_prompt(self, context:str,question:str) -> str :
        """prepares prompt based on type of question - factoid, causal or listing"""
        factoid = ["What", "Where", "When", "Explain", "Discuss", "Clarify"]
        causal = ["Why", "How"]
        listing = ["List", "Break down"]
        if any(word in question for word in factoid):
            prompt = """Generate an objective, formal and logically sound answer to this question, based on the given context. 
            The answer must spur curiosity, enable interactive discussions and make the user ask further questions. 
            It should be interesting and use advanced vocabulary and complex sentence structures.
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        elif any(word in question for word in causal):
            prompt = """Generate a procedural, knowledgeable and reasoning-based answer about this question, based on the given context. 
            The answer must use inference mechanisms and logic to subjectively discuss the topic. It should be creative and logic-oriented, analytical and extensive. 
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        elif any(word in question for word in listing):
            prompt = """Generate a list-type, descriptive answer to this question, based on the given context. 
            The answer should be very detailed and contain reasons, explanations and elaborations about the topic. It should be interesting and use advanced vocabulary and complex sentence structures. 
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        else:
            prompt = """Generate a detailed, interesting answer to this question, based on the given context. 
            The answer must be engaging and provoke interactions. It should use academic language and a formal tone. 
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        return prompt
    def answer_question(self,context:str,question:str,max_len:int = 300):
        prompt = self.prepare_prompt(context, question)
        # prompt = "Answer question from context:" + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        return self.text_gen(prompt,max_len).split("\nAnswer:")[1]
    
    def _text_gen_server(self,input_text:str,max_len:int=200):
        start_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input_text))
        results = self.model.generate_batch([start_tokens], max_length=max_len, sampling_topk=10,asynchronous=True)
        return results
    def _answer_question_server(self,context:str,question:str,max_len:int=300):       
        prompt = self.prepare_prompt(context, question)
        # return self._text_gen_server(prompt,max_len)
        
        # Cut off retrieved context within the max_len limit
        origin_context = prompt.split('Context : ')[1].split('\nQuestion:')[0]
        context_length = len(self.tokenizer.encode(origin_context))
        prompt_length = len(self.tokenizer.encode(prompt))
        fixed_prompt_length = prompt_length - context_length
        remain_avail_context_length = max_len - fixed_prompt_length - 2
        if context_length > remain_avail_context_length:
            new_encoded_context = self.tokenizer.encode(origin_context)[:remain_avail_context_length]
            new_context = self.tokenizer.decode(new_encoded_context)
            pmt, ctx = prompt.split('Context : ')
            ctx, qes = ctx.split('\nQuestion:')
            new_prompt = pmt + "Context : " + new_context + "\nQuestion:" + qes
            return self._text_gen_server(new_prompt,max_len)
        else:
            return self._text_gen_server(prompt,max_len)
    
    def answer_question_all(self,context_list,question:str,n_ans:int,max_len:int):
        ans_list = []
        if(self.deploy == "ct2-server"):
            temp_list = []
            for i in range(n_ans):
                temp_list.append(self._answer_question_server(context_list[i],question,max_len))
            while(False in [judge[0].done() for judge in temp_list]):
                pass
            
            for temp_ans in temp_list:
                ans_list.append(self.tokenizer.decode(temp_ans[0].result().sequences_ids[0]).split("\nAnswer:")[1])
                              
            return ans_list
        else:
            for i in range(n_ans):
                out_ans = self.answer_question(context_list[i],question,max_len)
                ans_list.append(out_ans)
            return ans_list
        
    def train_loss_ids(self,input_ids,label_ids):
        data = input_ids.to(self.device)
        outputs = self.model(data,labels = data)
        loss = outputs.loss.mean()
        return loss
    def save_checkpoint(self,saved_path):
        torch.save(
            self.model.state_dict(),
            saved_path,
        )
    def load_checkpoint(self,cp_path):
        cp = torch.load(cp_path)
        self.model.load_state_dict(cp)

# T5 model class
class seq2seq_model():
    def __init__(self,model_path = 'google/flan-t5-large',device = torch.device("cuda:1")):
        super(seq2seq_model,self).__init__()
        self.device = device 
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
    def generate(self,input_text:str,max_len:int = 100):
        input_ids = self.tokenizer(input_text,return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(input_ids,max_new_tokens = max_len)
        # outputs = self.model.generate(input_ids,penalty_alpha=0.6, top_k = 4,max_length=max_len)
        out_text = self.tokenizer.batch_decode(outputs,skip_special_tokens =True)[0]
        return out_text
    def answer_question(self,context:str,question:str,max_len:int = 100):
        prompt = "Answer question from context:\nContext:" + context.replace("\n"," ") + "\nQuestion:"+question.replace("\n"," ") + "\nAnswer:"
        return self.generate(prompt,max_len)
    def train_loss_ids(self,input_ids,label_ids):
        outputs = self.model(input_ids = input_ids.to(self.device),labels = label_ids.to(self.device))
        loss = outputs.loss
        return loss
    def train_loss_text(self,input_text,label_text):
        input_ids = self.tokenizer(input_text,return_tensors = 'pt').input_ids.to(self.device)
        labels = self.tokenizer(label_text,return_tensors = 'pt').input_ids.to(self.device)
        outputs = self.model(input_ids = input_ids, labels = labels)
        return outputs.loss
    def save_checkpoint(self,saved_path):
        torch.save(
            self.model.state_dict(),
            saved_path,
        )
    def load_checkpoint(self,cp_path):
        cp = torch.load(cp_path)
        self.model.load_state_dict(cp)

# GPT-J model class
class gpt_j():
    def __init__(self,model_path,device):
        super(gpt_j,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.gptj = GPTJForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    def text_gen(self,input_text:str,max_len:int = 200):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.gptj.generate(**inputs, pad_token_id = 50256,max_length = max_len,do_sample = True,early_stopping = True,temperature=0.8, top_p = 0.9)
        out_text = self.tokenizer.batch_decode(outputs)[0]
        return out_text
    
class benchmark_gptj():
    def __init__(self,model_path):
        super(benchmark_gptj,self).__init__()
        self.gptj = gpt_j(model_path)
        self.device = torch.device("cuda:0")
        self.dataset = load_dataset("coqa")
        
    def get_logger(self,filename, verbosity=1, name = None):
        level_dict = {0: logging.DEBUG, 1:logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename,"w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger
    
    def f_score(self,hypothesis,reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-l']['f']
    
    def eval(self,logger_path):
        logger = self.get_logger(logger_path)
        num_data = len(self.dataset['validation'])
        logger.info("Starting Evaluation")
        for pid in range(num_data):
            ans_list = []
            f_score_list = []
            context = self.dataset['validation'][pid]['story'].replace("\n"," ")
            n_word = len(context.split(" "))
            if(n_word>500):
                continue
            for num_his in range(len(self.dataset['validation'][pid]['questions'])-1):
                dialog_his = ""
                if(num_his!=0):
                    for i in range(num_his):
                        dialog_his = dialog_his + "\nquestion: "+self.dataset["validation"][pid]['questions'][i] + "\nanswer: " + self.dataset["validation"][pid]['answers']['input_text'][i]
                prompt = "Answer question from context\ncontext: "+context +  dialog_his +"\nquestion: " + self.dataset["validation"][pid]['questions'][num_his] + "\nanswer:"
                gt = self.dataset['validation'][pid]['answers']['input_text'][num_his]
                ans_text = self.gptj.text_gen(prompt,max_len=550)[0].split("\n")[num_his*2 +2 + 1]
                f_score = self.f_score(ans_text,gt)
                ans_list.append(ans_text)
                f_score_list.append(f_score)
            info_text = ""
            for i in range(len(ans_list)):
                info_text= info_text +  "QUESTION: " + self.dataset["validation"][pid]['questions'][i] + "\PREDICT:"+ ans_list[i] +"\nGROUNDTRUTH:" + self.dataset["validation"][pid]['answers']['input_text'][i] + "\nF-SCORE:" + str(f_score_list[i])+ "\n\n"
            info_text  = "\nCONTEXT:"+ context + "\n" + info_text 
            logger.info(info_text)
        return     
    
# Question Rewrite model class
class qr_model():
    def __init__(self,device):
        super(qr_model,self).__init__()
        # self.device = torch.device("cuda:0")
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained("castorini/t5-base-canard").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("castorini/t5-base-canard")
        self.sep = "|||"
    def qr(self,his_list,cur_q):
        prompt = ""
        for c in his_list:
            prompt += (c + self.sep)
        prompt += cur_q
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(inputs)
        out_text = self.tokenizer.batch_decode(outputs,skip_special_tokens = True)[0]
        return out_text

