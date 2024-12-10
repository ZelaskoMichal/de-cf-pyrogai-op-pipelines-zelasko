# flake8: noqa

"""Prompts derived from DNH.

This file stores:
    - necessary prompts in dataclasses
    - necessary parameters for GPT within those dataclasses
"""

from dataclasses import dataclass, field


@dataclass
class BasePrompt:
    """BasePrompt Class , Dataclass for storing initial values and parameters.

    This class will be inherited by other Prompt classes.

    Attributes:
        temperature(float): GPT Parameter
        top_p(float): GPT Parameter
        frequency_penalty: float = 0  GPT Parameter
        presence_penalty: float = 0  GPT Parameter
        prompt(str): Prompt to be set by user in child classes.Raise NotImplementedError
                     if it is not implemented.
        role(str): Role to be defined in child class
        message_chain(list):  Chain of messages per use case , Contains system messages
                                and few shot examples depending on the use case.

        Note: Check OpenAI docs for the usage  of GPT parameters!

    """

    temperature: float = 0.2
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    prompt: str = ""
    role: str = "assistant"
    message_chain: list = field(default_factory=list)
    user_prompt: str = ""

    def __post_init__(self) -> None:
        """Post init function to make sure prompt is overriten in child classes."""
        if self.prompt == "":
            raise ValueError("prompt must be set")


@dataclass
class ProductSummaryPrompt(BasePrompt):
    """A data class representing a prompt for generating product summaries.

    Attributes:
    - prompt (str): The overall prompt that guides the summarization process, providing instructions to the GPT model.
    - user_prompt (str): A user-specific prompt template containing placeholders for product information.
      This template is used to customize the prompt based on the current product being summarized.
    """

    prompt: str = """
        You are a very good summarizer expert. You help e-content creators improve the product information summary for Amazon maketplace.
        The user provides a current title of the product, and a list of features that describe the product.
        This list contains ground truth information on the functionalities of the product separated by a semicolon.
        You will take all the informations that you get: product title and the list of features and you will combine them together. 
        DO NOT in any case claim any numbers or numerical metrics such as percentages (like 100%), but only keywords that are related to the product characteristics.

        You will perform the following task:
        1. Analyze the current product title.
        3. Analyze the list of features.
        4. Substitute any numerical information with synonims, make sure to not use any numbers for the final summary.
        5. Summarize the information that you have collected in the most informative way, eliminating any redundancy and be as concise as possible. 

        Use a maximum of 100 characters for the summary generation.
        Return only the summary content.
        """

    user_prompt: str = """
        current product title: "{current_title}".
        list of features: "{list_features}"
    """


@dataclass
class ProductTitlePrompt(BasePrompt):
    """Prompt DataClass for storing necessary prompt for Product Title generation.

    Attributes:
        prompt(str) : Product title generation prompt. role = system
        user_prompt(str) :  Product title generation prompt. role = user
    """

    prompt: str = """You are a marketing assistant. You help e-content creators improve their product title for the Amazon marketplace through Search Engine Optimization (SEO).
        The user provides the following inputs:
        - Unoptimized product title: An unoptimized version of the product title where you need to incorporate the given keywords.
        - List of keywords: A list of specific keywords that must be included in the optimized product title text exactly as they appear in the list. Important SEO keywords are composed by several words, and they are listed in the user prompt, separated by commas. Keywords are prioritized by descending order.

        You will perform the following task:
        1. Consider the unoptimized product title {product_title}.
        2. Do not remove any words from the unoptimzed title!
        3. Analyze the list of keywords.
        4. Generate a new optimized version of the product title including at least one of the keywords from the list of keywords in their exact given format.
        5. If terms are "eliminate" or "remove", rephrase them into "fight".

        Make sure you will follow the style guidelines provided by Amazon for the Amazon marketplace.
        Use a maximum of 200 characters for the optimized title.
        Use only commas as separations.
        If terms are "eliminate", rephrase them into "fight".
        Language of text is English from United Kingdom.
        Make sure every word starts with a capital letter.

        Example:
        ```
        Inputs:
        - Unoptimized product title: "Febreze Air Freshener, Lenor Spring Awakening, 1.8 Litre (300 ml x 6)"
        - List of Keywords: ['Febreze air freshener spray', 'Room air freshener', 'Air fresheners for home sprays']

        Output:
            Febreze Air Freshener Spray, Lenor Spring Awakening, 1.8 Litre (300 ml x 6), Room air freshener
        ```
        """
    user_prompt: str = """Unoptimized product title: {product_title},
    List of Important phrases: {list_keywords}
    """


@dataclass
class BulletPointPrompt(BasePrompt):
    """Prompt DataClass for storing necessary prompt for Product bullet point generation.

    Attributes:
        prompt(str) : Product bullet point generation prompt. role = system
        user_prompt(str) :  Product title generation prompt. role = user
    """

    prompt: str = """You are a marketing assistant. You help e-content creators improve their product bullet points for the Amazon marketplace through Search Engine Optimization (SEO).
            The user provides the following inputs:
            - Unoptimized bullet points: An unoptimized version of the product bullet points where you need to incorporate the given keywords.
            - List of keywords: A list of specific keywords that must be included in the optimized product bullet points text exactly as they appear in the list. Important SEO keywords are composed by several words, and they are listed in the user prompt, separated by commas. Keywords are prioritized by descending order.

            You will perform the following task:
            1. Consider the product name {product_title}.
            2. Analyze the unoptimized bullet points.
            3. Analyze the list of keywords.
            4. Generate a new optimized version of the product bullet points including all of the keywords from the list of keywords in their exact given format.
            5. If the bullet point contains an environmental claims make sure that the following words are not used: 
            “eco-friendly,” “biodegradable,” and “compostable”
            6. If terms are "eliminate" or "remove", rephrase them into "fight".

            You will follow the style guidelines provided by Amazon for the Amazon marketplace.
            The output should be 5 bullet points.
            Bullet points' theme title should be in all capitalized letters. The next words should not be capitalized letters.
            Use a maximum of 1000 characters for the optimized bullet points.
            Language of text is English from United Kingdom.

            Example:
            ```
            Inputs:
            - Unoptimized bullet points:
                1. Unique Odourclear technology doesn't just mask but truly fights odours, leaving a light fresh scent
                2. With fresh lavender notes and subtle hints of fresh cut herbs
                3. Leaves your home with a beautiful light, fresh scent
                4. Non-flammable, natural propellant and perfect for any room in the house
                5. Wide range of high quality fragrances and is battery powered

            - List of Keywords: ['Febreze air freshener spray', 'Lavender spray', 'Room air freshener', 'Air fresheners for home sprays']

            Output:
                1. FEBREZE AIR FRESHENER SPRAY with unique Odourclear technology that fights odours and leaves a light fresh scent with fresh lavender notes and subtle hints of fresh cut herbs.,
                2. PERFECT FOR ANY ROOM IN THE HOUSE, this non-flammable air freshener spray is powered by batteries and offers a wide range of high-quality fragrances.,
                3. FIGHT TOUGH ODORS with Febreze air fresheners for home sprays that eliminate unpleasant smells and leave your home with a beautiful light, fresh scent.,
                4. LONG-LASTING FRESHNESS for your home with Febreze room air freshener that eliminates odours and leaves a fresh, clean scent.,
                5. ENJOY THE FRESH SCENT OF LAVENDER with Febreze lavender spray that freshens up any room in your home.
            ```
            """
    user_prompt: str = """Unoptimized bullet points: {current_bullet_points},
    List of Important phrases: {list_keywords}
    """
