import random
from collections import defaultdict
import re

class MarkovChainGenerator:
    def __init__(self, order=2, level='word'):
        """
        Initialize the Markov Chain generator
        
        Args:
            order (int): The number of previous words/chars to consider
            level (str): 'word' or 'char' for word-level or character-level generation
        """
        self.order = order
        self.level = level
        self.model = defaultdict(list)
        
    def tokenize(self, text):
        """Convert text into tokens based on level"""
        if self.level == 'word':
            # Split text into words, preserving basic punctuation
            return re.findall(r"[\w']+|[.,!?;]", text)
        else:
            # For character level, just return the characters
            return list(text)
    
    def train(self, text):
        """Train the Markov Chain model on the input text"""
        tokens = self.tokenize(text)
        
        # Create sequences of tokens
        for i in range(len(tokens) - self.order):
            # Create state tuple (previous tokens)
            state = tuple(tokens[i:i + self.order])
            # Get next token
            next_token = tokens[i + self.order]
            # Add to model
            self.model[state].append(next_token)
        
        print(f"Model trained on {len(tokens)} tokens with {len(self.model)} states")
    
    def generate(self, length=100, seed=None):
        """
        Generate new text using the trained model
        
        Args:
            length (int): Number of tokens to generate
            seed (str): Optional starting text
        """
        if not self.model:
            return "Error: Model not trained yet!"
        
        # Start with a random state from the model if no seed is provided
        if seed:
            current_state = tuple(self.tokenize(seed)[-self.order:])
            if current_state not in self.model:
                current_state = random.choice(list(self.model.keys()))
        else:
            current_state = random.choice(list(self.model.keys()))
        
        result = list(current_state)
        
        # Generate new tokens
        for _ in range(length):
            if current_state not in self.model:
                # If we reach a dead end, pick a new random state
                current_state = random.choice(list(self.model.keys()))
            
            # Get next token
            next_token = random.choice(self.model[current_state])
            result.append(next_token)
            
            # Update state
            current_state = tuple(result[-self.order:])
        
        # Join tokens based on level
        if self.level == 'word':
            return self._format_word_output(result)
        else:
            return ''.join(result)
    
    def _format_word_output(self, words):
        """Format word-level output with proper spacing"""
        text = ''
        for i, word in enumerate(words):
            if word in '.,!?;':
                text = text.rstrip() + word + ' '
            else:
                text += word + ' '
        return text.strip()

def main():
    # Example texts for training
    example_texts = {
        1: """The quick brown fox jumps over the lazy dog. 
             The dog sleeps peacefully while the fox continues to jump.""",
        2: """To be or not to be, that is the question.
             Whether 'tis nobler in the mind to suffer.""",
        3: """In the beginning God created the heaven and the earth.
             And the earth was without form, and void."""
    }
    
    # Print available example texts
    print("Available example texts:")
    for i, text in example_texts.items():
        print(f"\n{i}. {text[:50]}...")
    
    # Get user input
    choice = input("\nChoose an example text (1-3) or enter 'custom' to input your own: ")
    
    if choice.lower() == 'custom':
        training_text = input("\nEnter your training text: ")
    else:
        training_text = example_texts[int(choice)]
    
    # Get generation parameters
    level = input("\nChoose generation level (word/char): ").lower()
    order = int(input("Enter the order (1-3 recommended): "))
    length = int(input("Enter the desired length (number of tokens): "))
    seed = input("Enter a seed text (optional, press Enter to skip): ")
    
    # Create and train the model
    generator = MarkovChainGenerator(order=order, level=level)
    generator.train(training_text)
    
    # Generate text
    generated_text = generator.generate(length=length, seed=seed if seed else None)
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
