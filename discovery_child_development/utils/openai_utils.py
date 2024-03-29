import asyncio
import json
import logging
import string

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import os
import dotenv
import aiofiles
from openai import OpenAI, AsyncOpenAI

from discovery_child_development.utils.utils import current_time

dotenv.load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

logger = logging.getLogger(__name__)


def print_prompt(prompt: list):
    """Neatly prints your prompt messages"""
    for m in prompt:
        print(f"{m['role']}: {m['content']}\n")


@dataclass
class BasePromptTemplate(ABC):
    """Base template prompts flexibly."""

    initial_template: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Keep the initial template."""
        self.initial_template = self._initialize_template()

    @abstractmethod
    def _initialize_template(self) -> None:
        """To be implemented by child classes"""
        pass

    @staticmethod
    @abstractmethod
    def _from_dict(data: Dict) -> None:
        """Create a Template instance from a dictionary."""
        pass

    def format_message(self, **kwargs) -> None:
        """Process a message and fill in any placeholders."""

        def recursive_format(value: Union[str, dict]) -> Union[str, dict]:
            if isinstance(value, str):
                placeholders = self._extract_placeholders(value)
                if placeholders:
                    return value.format(**kwargs)
                return value
            elif isinstance(value, dict):
                return {k: recursive_format(v) for k, v in value.items()}
            else:
                return value

        for k in self.__dict__.keys():
            if k != "initial_template":
                self.__dict__[k] = recursive_format(self.initial_template[k])

    @classmethod
    def load(cls, obj: Union[Dict, str]) -> "BasePromptTemplate":
        """Load a Template instance from a JSON file or a dictionary."""
        if isinstance(obj, str):
            return cls._from_json(obj)
        elif isinstance(obj, Dict):
            return cls._from_dict(obj)
        else:
            raise TypeError(
                f"Expected a JSON file path or a dictionary, got {type(obj)}."
            )

    @staticmethod
    def _exclude_keys(
        d: dict,
        exclude: Optional[List[str]] = None,  # noqa: B006
    ) -> dict:
        """Exclude keys from a dictionary."""
        if not d["name"]:
            d.pop("name", None)

        if exclude:
            for item in exclude:
                d.pop(item, None)
            return d
        return d

    def to_prompt(
        self,
        exclude: Optional[List[str]] = ["initial_template"],  # noqa: B006
    ) -> Dict:
        """Convert a Template instance to a JSON string."""
        d = asdict(self)
        return self._exclude_keys(d, exclude=exclude)

    @staticmethod
    def _extract_placeholders(s: str) -> List[str]:
        """Extract placeholder variables that can be filled in an f-string."""
        formatter = string.Formatter()
        return [
            field_name
            for _, field_name, _, _ in formatter.parse(s)
            if field_name is not None
        ]

    @classmethod
    def _from_json(cls, json_path: str) -> "BasePromptTemplate":
        """Create a Template instance by providing a JSON path."""
        return cls._from_dict(cls._read_json(json_path))

    @staticmethod
    def _read_json(json_path: str) -> Dict:
        """Read a JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    def to_json(self, path: str) -> None:
        """Convert a Template instance to a JSON string."""
        self._write_json(self.initial_template, path)

    def _write_json(self, data: Dict, path: str) -> None:
        """Write a JSON file."""
        with open(path, "w") as f:
            json.dump(data, f)


@dataclass
class MessageTemplate(BasePromptTemplate):
    """Create a template for a message prompt."""

    role: str
    content: str
    name: Optional[str] = None

    def __post_init__(self) -> None:
        """Keep the initial template and error when the role is function but not name was given."""
        super().__post_init__()
        if self.role == "function" and not self.name:
            raise ValueError(
                "The 'name' attribute is required when 'role' is 'function'."
            )

    def _initialize_template(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content, "name": self.name}

    @staticmethod
    def _from_dict(data: Dict) -> "MessageTemplate":
        instance = MessageTemplate(**data)
        # Validate after initialisation
        if instance.role == "function" and not instance.name:
            raise ValueError(
                "The 'name' attribute is required when 'role' is 'function'."
            )
        return instance


@dataclass
class FunctionTemplate(BasePromptTemplate):
    """Create a template for an OpenAI function."""

    name: str
    description: str
    parameters: Dict[
        str, Union[str, Dict[str, Dict[str, Union[str, List[str]]]], List[str]]
    ]

    def _initialize_template(
        self,
    ) -> Dict[str, Union[str, Dict[str, Dict[str, Union[str, List[str]]]], List[str]]]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @staticmethod
    def _from_dict(data: Dict) -> "FunctionTemplate":
        """Create a Template instance from a dictionary."""
        return FunctionTemplate(**data)


class Classifier:
    """Classify text."""

    @classmethod
    def generate(
        cls,
        messages: List[Union[str, Dict]],
        message_kwargs: Optional[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **openai_kwargs,
    ) -> Union[Dict, str]:
        """Generate text using OpenAI's API.

        More details on the API and messages: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Args:
            messages
                A list of messages to send to the API. They can be:
                - dictionaries
                - str (JSON file path)
                - instances of classes that inherit from BasePromptTemplate

            message_kwargs
                A dictionary of keyword arguments to pass to the messages.

            model
                The OpenAI model to use.

            temperature
                The sampling temperature.

            openai_kwargs
                Keyword arguments to pass to the OpenAI API.

        Returns:
            A dictionary containing the response from the API.

        """
        if not message_kwargs:
            message_kwargs = {}

        messages = [
            cls.prepare_message(message, **message_kwargs) for message in messages
        ]

        response = cls._call(
            messages=messages,
            temperature=temperature,
            model=model,
            **openai_kwargs,
        )

        parsed_response = json.loads(
            response.choices[0].message.function_call.arguments
        )
        if parsed_response:
            parsed_response["id"] = message_kwargs["id"]
            return parsed_response

        return message_kwargs["id"]

    @classmethod
    def prepare_message(cls, obj: Union[MessageTemplate, dict, str], **kwargs) -> Dict:
        """Process a message."""
        if not isinstance(obj, MessageTemplate):
            prompt = MessageTemplate.load(obj)
        else:
            prompt = obj

        prompt.format_message(**kwargs)

        return prompt.to_prompt()

    def _call(
        messages: List[Dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        response = client.chat.completions.create(
            messages=messages, model=model, temperature=temperature, **kwargs
        )

        return response  # type: ignore

    @classmethod
    async def agenerate(
        cls,
        messages: List[Union[str, Dict]],
        message_kwargs: Optional[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        concurrency: int = 10,
        **openai_kwargs,
    ) -> Dict:
        """Generate text using async OpenAI's API.

        More details on the API and messages: https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Args:
            messages
                A list of messages to send to the API. They can be:
                - dictionaries
                - str (JSON file path)

            message_kwargs
                A dictionary of keyword arguments to pass to the messages.

            temperature
                The sampling temperature.

            openai_kwargs
                Keyword arguments to pass to the OpenAI API.

            concurrency:
                The number of concurrent requests to make.

        Returns:
            A dictionary containing the response from the API.

        """
        semaphore = asyncio.Semaphore(concurrency)
        async with semaphore:
            if not message_kwargs:
                message_kwargs = {}

            messages = [
                cls.prepare_message(message, **message_kwargs) for message in messages
            ]

            response = await cls._acall(
                messages=messages,
                temperature=temperature,
                model=model,
                **openai_kwargs,
            )
            response = response.choices[0].message.function_call.arguments
            parsed_response = await cls._parse_json(response)
            if parsed_response:
                parsed_response["id"] = message_kwargs["id"]
                parsed_response["source"] = message_kwargs["source"]
                parsed_response["text"] = message_kwargs["text"]
                parsed_response["model"] = model
                parsed_response["timestamp"] = current_time()
                return parsed_response

            return message_kwargs["id"]

    async def _acall(
        messages: List[Dict],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_retries: int = 5,
        retry_delay: int = 10,
        **kwargs,
    ) -> Dict:
        for attempt in range(max_retries):
            try:
                response = await aclient.chat.completions.create(
                    messages=messages, model=model, temperature=temperature, **kwargs
                )
                return response  # type: ignore
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise e

    @staticmethod
    async def _try_parse_json(item: str) -> Union[dict, None]:
        try:
            return json.loads(item)
        except json.JSONDecodeError as e:
            return e

    @staticmethod
    async def _parse_json(item: str) -> Union[dict, None]:
        result = await Classifier._try_parse_json(item)
        if isinstance(result, json.JSONDecodeError):
            result = await Classifier._try_parse_json(item.replace("'", '"'))
            if isinstance(result, json.JSONDecodeError):
                logging.error(f"Invalid JSON: Error: {str(result)}")
                return None
        return result

    @staticmethod
    async def write_line_to_file(
        item: dict, path: str, filename: str = "parsed_json"
    ) -> None:
        """Write the item to a file."""
        file = f"{path}/{filename}_invalid.txt"
        if isinstance(item, dict):
            file = f"{path}/{filename}.jsonl"

        async with aiofiles.open(file, "a") as f:
            await f.write(f"{json.dumps(item)}\n")
