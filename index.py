import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tkinter as tk
from tkinter import scrolledtext
import threading

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        print("Initializing GPT-2 model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        print("Model loaded successfully!")

    def generate_text(self, prompt, max_length=200, temperature=0.7):
        # Encode the input prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

class TextGeneratorGUI:
    def __init__(self):
        self.generator = TextGenerator()
        self.setup_gui()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("GPT-2 Text Generator")
        self.window.geometry("800x600")

        # Prompt input
        tk.Label(self.window, text="Enter your prompt:").pack(pady=5)
        self.prompt_input = scrolledtext.ScrolledText(self.window, height=4)
        self.prompt_input.pack(padx=10, pady=5, fill=tk.X)

        # Parameters frame
        params_frame = tk.Frame(self.window)
        params_frame.pack(pady=5)

        # Max length slider
        tk.Label(params_frame, text="Max Length:").grid(row=0, column=0, padx=5)
        self.max_length_var = tk.IntVar(value=200)
        self.max_length_slider = tk.Scale(params_frame, from_=50, to=500, 
                                        orient=tk.HORIZONTAL, variable=self.max_length_var)
        self.max_length_slider.grid(row=0, column=1, padx=5)

        # Temperature slider
        tk.Label(params_frame, text="Temperature:").grid(row=1, column=0, padx=5)
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_slider = tk.Scale(params_frame, from_=0.1, to=1.0, 
                                         orient=tk.HORIZONTAL, resolution=0.1, 
                                         variable=self.temperature_var)
        self.temperature_slider.grid(row=1, column=1, padx=5)

        # Generate button
        self.generate_button = tk.Button(self.window, text="Generate", command=self.generate)
        self.generate_button.pack(pady=10)

        # Output text
        tk.Label(self.window, text="Generated Text:").pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(self.window, height=15)
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    def generate(self):
        self.generate_button.config(state=tk.DISABLED)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Generating...")
        
        # Run generation in a separate thread
        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        prompt = self.prompt_input.get("1.0", tk.END).strip()
        try:
            generated_text = self.generator.generate_text(
                prompt,
                max_length=self.max_length_var.get(),
                temperature=self.temperature_var.get()
            )
            self.window.after(0, self._update_output, generated_text)
        except Exception as e:
            self.window.after(0, self._update_output, f"Error: {str(e)}")

    def _update_output(self, text):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.generate_button.config(state=tk.NORMAL)

    def run(self):
        self.window.mainloop()

def main():
    print("Select mode:")
    print("1. Command Line Interface")
    print("2. Graphical User Interface")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        generator = TextGenerator()
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            
            print("\nGenerating text...")
            generated_text = generator.generate_text(prompt)
            print("\nGenerated Text:")
            print(generated_text)
    
    elif choice == "2":
        gui = TextGeneratorGUI()
        gui.run()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
