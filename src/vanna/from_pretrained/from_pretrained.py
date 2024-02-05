from base import VannaBase

from gpt4all import GPT4All

class LLMFromPretrained(VannaBase):
    def __init__(self, path_to_weights, n_threads, response_context_length, device = 'gpu', config=None):
        super().__init__(config)

        self.llm = GPT4All(model_name=path_to_weights, n_threads=n_threads, n_ctx=response_context_length, verbose=True, device=device)
    
    def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        pass
        
    def get_followup_questions_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list, **kwargs):
        pass
    
    def get_sql_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list, **kwargs):
        pass

    def submit_prompt(self, prompt, **kwargs) -> str:
        return self.llm.generate(prompt=prompt)
