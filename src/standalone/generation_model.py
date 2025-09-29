from google import genai
from prompt_optimization.client.google_genai_client import init_google_genai_client
import time
import asyncio
import nest_asyncio


class GenerationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = init_google_genai_client()
        self.default_config = {
            "max_output_tokens": 8192,
            "temperature": 0.01,
            "top_p": 0.95,
        }
        self.default_diversity_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 1,
        }
        self.api_calls = 0
        self.total_tokens = 0
        self.total_time = 0
        self.llm_query_retries = 10
        self.async_timeout = 180
        # patch the jupyter event loop for async gemini requests
        nest_asyncio.apply()

    def ask_doc(self, prompt, doc, gen_config, schema=None) -> str:
        if schema:
            new_config = gen_config
            new_config["response_mime_type"] = "application/json"
            new_config["response_schema"] = schema
            gen_config = new_config
        config = genai.types.GenerateContentConfig(**gen_config)
        contents = [doc, prompt]
        response = self.__query(contents=contents, config=config)
        if schema:
            return response.parsed
        return response.text

    def ask(self, prompt, gen_config, schema) -> str:
        if schema:
            new_config = gen_config
            new_config["response_mime_type"] = "application/json"
            new_config["response_schema"] = schema
            gen_config = new_config
        config = genai.types.GenerateContentConfig(**gen_config)
        response = self.__query(contents=prompt, config=config)
        if schema:
            return response.parsed
        return response.text

    def ask_with_batch_docs(self, prompt, gen_config, docs, schema):
        if schema:
            new_config = gen_config
            new_config["response_mime_type"] = "application/json"
            new_config["response_schema"] = schema
            gen_config = new_config
        config = genai.types.GenerateContentConfig(**gen_config)
        contents = []
        for i, doc in enumerate(docs):
            contents.append(f"Example {i}: ")
            contents.append(doc)
        contents.append(prompt)
        response = self.__query(contents=contents, config=config)
        if schema:
            return response.parsed
        return response.text

    def update_total_tokens(self, response):
        self.total_tokens += response.usage_metadata.total_token_count

    def __query(self, contents, config):
        # patch the jupyter event loop
        nest_asyncio.apply()

        async def _generate():
            return await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                ),
                timeout=self.async_timeout,
            )

        for retry in range(self.llm_query_retries):
            try:
                start_time = time.perf_counter()
                response = asyncio.get_event_loop().run_until_complete(_generate())
                end_time = time.perf_counter()
                self.update_total_tokens(response)
                self.api_calls += 1
                self.total_time += end_time - start_time
                break
            except Exception as error:
                print(
                    f"Api exception, retrying up to {self.llm_query_retries} times",
                    error,
                )
        else:
            raise Exception("The program failed multiple times, aborting.")
        return response

    @classmethod
    def create_model(cls, model_name):
        return cls(model_name=model_name)

    def query_llm(self, prompt_text: str, config=None, doc=None, schema=None):
        if config is None:
            config = self.default_config
        if doc is None:
            return self.ask(prompt=prompt_text, gen_config=config, schema=schema)
        elif isinstance(doc, list):
            if len(doc) == 0:
                raise ValueError("No examples passed to the model")
            return self.ask_with_batch_docs(
                prompt=prompt_text, docs=doc, gen_config=config, schema=schema
            )
        elif isinstance(doc, str):
            return self.ask(prompt=f"{prompt_text}. Input: {doc}.", gen_config=config)
        else:
            return self.ask_doc(
                prompt=prompt_text, doc=doc, gen_config=config, schema=schema
            )

    def query_llm_test(
        self,
        prompt_text: str,
        config=None,
        doc=None,
        schema=None,
        logger=None,
        doc_id=None,
    ):
        if config is None:
            config = self.default_config
        if doc is None:
            return self.ask(prompt=prompt_text, gen_config=config)
        elif isinstance(doc, list):
            if len(doc) == 0:
                raise ValueError("No examples passed to the model")
            result = self.ask_with_batch_docs(
                prompt=prompt_text, docs=doc, gen_config=config, schema=schema
            )
            logger.custom(f"API response: {result}")
            return result
        elif isinstance(doc, str):
            return self.ask(prompt=f"{prompt_text}. Input: {doc}.", gen_config=config)
        else:
            result = self.ask_doc(
                prompt=prompt_text, doc=doc, gen_config=config, schema=schema
            )
            logger.custom(f"Prompt: {prompt_text}")
            logger.custom(f"API response. Doc id: {doc_id}, response: {result}")
            return result
