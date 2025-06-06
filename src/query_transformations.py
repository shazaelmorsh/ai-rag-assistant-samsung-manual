"""Module for transforming and enhancing user queries for Samsung phone support.

This module provides functionality to normalize, expand, and specify user queries
to improve the quality of responses in a Samsung phone support context.
"""

from typing import List
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class QueryTransformer:
    """A class for transforming and enhancing user queries about Samsung phones.
    
    This class provides methods to normalize technical terms, generate alternative
    phrasings, add technical specificity, and extract key concepts from queries.
    """
    
    def __init__(self, llm=None):
        """Initialize QueryTransformer with optional language model.
        
        Args:
            llm: Language model for query transformation (defaults to OpenAI with temperature=0)
        """
        self.llm = llm or ChatOpenAI(
            temperature=0,
            model="openai/gpt-4o-mini-2024-07-18",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/shazaelmorsh/ai-rag-assistant-samsung-manual",
                "X-Title": "Samsung AI RAG Assistant"
            }
        )
        
        self.expansion_prompt = PromptTemplate(
            template="""You are a Samsung phone support expert. Given a user's question, generate 3 alternative ways to ask the same question. 
            
            Original question: {question}
            
            Generate alternatives in a list format, maintaining the same intent but using different phrasings.
            Do not include explanations or any other text, just the list of questions.""",
            input_variables=["question"]
        )
        
        self.specification_prompt = PromptTemplate(
            template="""You are a Samsung phone support expert. Given a user's question, make it more specific by adding relevant technical details 
            and Samsung-specific terminology. Focus on:
            - Specific feature names
            - UI elements
            - Settings locations
            - Common troubleshooting steps
            
            Original question: {question}
            
            Provide a more detailed, technically-specific version of the question.
            Do not include explanations or any other text, just the enhanced question.""",
            input_variables=["question"]
        )

    def normalize_query(self, query: str) -> str:
        """Normalize technical terms and remove noise from the query.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Normalized query with standardized technical terms
        """
        replacements = {
            r"(?i)one\s*ui": "One UI",
            r"(?i)samsung\s*account": "Samsung Account",
            r"(?i)galaxy\s*store": "Galaxy Store",
            r"(?i)s\s*pen": "S Pen",
            r"(?i)smart\s*switch": "Smart Switch",
            r"(?i)good\s*lock": "Good Lock",
            r"(?i)secure\s*folder": "Secure Folder"
        }
        
        normalized = query
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        return normalized

    def expand_query(self, query: str) -> List[str]:
        """Generate alternative phrasings of the query.
        
        Args:
            query (str): Original user query
            
        Returns:
            List[str]: List of alternative phrasings including the original query
        """
        response = self.llm.invoke(
            self.expansion_prompt.format(question=query)
        )
        alternatives = [q.strip() for q in response.strip().split('\n') if q.strip()]
        all_queries = [query] + alternatives
        return list(dict.fromkeys(all_queries))

    def specify_query(self, query: str) -> str:
        """Make the query more specific with technical details.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Enhanced query with added technical specificity
        """
        response = self.llm.invoke(
            self.specification_prompt.format(question=query)
        )
        return response.strip()

    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract key technical concepts and features from the query.
        
        Args:
            query (str): User query to analyze
            
        Returns:
            List[str]: List of unique technical concepts found in the query
        """
        concept_patterns = [
            r"(?i)(?:Android|One UI)\s*\d+(?:\.\d+)*",
            r"(?i)Galaxy\s+[A-Za-z]\d+(?:\s+[A-Za-z]+)*",
            r"(?i)(?:camera|battery|security|display|storage|network|wifi|bluetooth|fingerprint|face recognition|wireless charging|dex|s pen)\b",
            r"(?i)(?:settings|menu|app|notification|widget)\b",
            r"(?i)(?:backup|restore|update|sync|reset|pair|connect)\b"
        ]
        
        concepts = []
        for pattern in concept_patterns:
            matches = re.finditer(pattern, query)
            concepts.extend(match.group(0) for match in matches)
        
        return list(dict.fromkeys(concepts))

    def transform_query(self, query: str) -> dict:
        """Apply all query transformations and return results.
        
        Args:
            query (str): Original user query
            
        Returns:
            dict: Dictionary containing all transformed versions of the query
        """
        normalized = self.normalize_query(query)
        expanded = self.expand_query(normalized)
        specified = self.specify_query(normalized)
        concepts = self.extract_key_concepts(normalized)
        
        return {
            "original": query,
            "normalized": normalized,
            "expanded": expanded,
            "specified": specified,
            "key_concepts": concepts
        } 