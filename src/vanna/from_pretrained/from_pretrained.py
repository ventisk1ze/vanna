from base import VannaBase

from gpt4all import GPT4All

class LLMFromPretrained(VannaBase):
    def __init__(self, path_to_weights, n_threads, response_context_length, device = 'gpu', config=None):
        super().__init__(config)

        self.llm = GPT4All(model_name=path_to_weights, n_threads=n_threads, n_ctx=response_context_length, verbose=True, device=device, allow_download=False)
        self.prompt_template = 'USER: {0}\nASSISTANT: '
    
    def generate_plotly_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        system_prompt = """
        The user will give you SQL and you will try to guess what the business question this query is answering. 
        Return just the question without any additional explanation. Do not reference the table name in the question.
        """
        
        return self.submit_prompt(
            self._clear_string(system_prompt) + self.prompt_template.format(sql)
        )
        
    def get_followup_questions_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list, **kwargs):
        pass
    
    def get_sql_prompt(self, question: str, question_sql_list: list, ddl_list: list, doc_list: list, **kwargs):
        system_prompt = """
        The user provides a question and you provide SQL. You will only respond with SQL code and not with any explanations.
        Respond with only SQL code. Do not answer with any explanations -- just the code.
        """

    def submit_prompt(self, prompt, **kwargs) -> str:
        return self.llm.generate(prompt=prompt)
    
    @staticmethod
    def _clear_string(string) -> str:
        return string.strip().replace(r'\n', '').replace(r'\r', '').replace(r'\t', '')
