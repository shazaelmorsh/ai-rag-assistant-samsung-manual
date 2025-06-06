from typing import List
import re
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

class QueryTransformer:
    def __init__(self, llm=None):
        self.llm = llm or OpenAI(temperature=0)
        
        # Prompt template for query expansion
        self.expansion_prompt = PromptTemplate(
            template="""You are a Samsung phone support expert. Given a user's question, generate 3 alternative ways to ask the same question. 
            
            Original question: {question}
            
            Generate alternatives in a list format, maintaining the same intent but using different phrasings.
            Do not include explanations or any other text, just the list of questions.""",
            input_variables=["question"]
        )
        
        # Prompt template for query specification
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
        """Normalize the query by standardizing technical terms and removing noise."""
        # Standardize common variations
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
            
        # Remove excessive punctuation and whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        return normalized

    def expand_query(self, query: str) -> List[str]:
        """Generate alternative phrasings of the query."""
        response = self.llm.invoke(
            self.expansion_prompt.format(question=query)
        )
        # Extract questions from response
        alternatives = [q.strip() for q in response.strip().split('\n') if q.strip()]
        # Add original query and return unique set
        all_queries = [query] + alternatives
        return list(dict.fromkeys(all_queries))  # Remove duplicates while preserving order

    def specify_query(self, query: str) -> str:
        """Make the query more specific with technical details."""
        response = self.llm.invoke(
            self.specification_prompt.format(question=query)
        )
        return response.strip()

    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract key technical concepts and features from the query."""
        # Common Samsung phone-related concepts
        concept_patterns = [
            r"(?i)(?:Android|One UI)\s*\d+(?:\.\d+)*",  # OS versions
            r"(?i)Galaxy\s+[A-Za-z]\d+(?:\s+[A-Za-z]+)*",  # Phone models
            r"(?i)(?:camera|battery|security|display|storage|network|wifi|bluetooth|fingerprint|face recognition|wireless charging|dex|s pen)\b",  # Features
            r"(?i)(?:settings|menu|app|notification|widget)\b",  # UI elements
            r"(?i)(?:backup|restore|update|sync|reset|pair|connect)\b"  # Actions
        ]
        
        concepts = []
        for pattern in concept_patterns:
            matches = re.finditer(pattern, query)
            concepts.extend(match.group(0) for match in matches)
        
        return list(dict.fromkeys(concepts))  # Remove duplicates while preserving order

    def transform_query(self, query: str) -> dict:
        """Apply all query transformations and return results."""
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