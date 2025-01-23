import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import os

class StyleTransfer:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained VGG19 model
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Define content and style layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Image processing
        self.loader = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device)
    
    def save_image(self, tensor, path):
        """Save the generated image"""
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )(image)
        image = (image.clamp(0, 1) * 255).numpy().transpose(1, 2, 0).astype('uint8')
        Image.fromarray(image).save(path)
    
    def get_features(self, image):
        """Extract features from image using VGG19"""
        features = {}
        x = image
        layer_count = {'conv': 1, 'relu': 1, 'pool': 1}
        
        for name, layer in self.model._modules.items():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                layer_name = f'conv_{layer_count["conv"]}'
                features[layer_name] = x
                layer_count['conv'] += 1
            
        return features
    
    def gram_matrix(self, tensor):
        """Calculate Gram Matrix"""
        _, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        return torch.mm(tensor, tensor.t())
    
    def transfer_style(self, content_path, style_path, num_steps=300,
                      content_weight=1, style_weight=1e6):
        """Perform neural style transfer"""
        print("Loading images...")
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        # Generate input image
        input_img = content_img.clone()
        
        # Setup optimizer
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        print("Extracting features...")
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)
        
        # Calculate style gram matrices
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers
        }
        
        print("Starting style transfer...")
        step = [0]
        
        def closure():
            optimizer.zero_grad()
            features = self.get_features(input_img)
            
            # Content loss
            content_loss = torch.mean((features[self.content_layers[0]] - 
                                     content_features[self.content_layers[0]]) ** 2)
            
            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                input_gram = self.gram_matrix(features[layer])
                style_gram = style_grams[layer]
                layer_style_loss = torch.mean((input_gram - style_gram) ** 2)
                style_loss += layer_style_loss / len(self.style_layers)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            
            step[0] += 1
            if step[0] % 50 == 0:
                print(f"Step {step[0]}/{num_steps}")
            
            return total_loss
        
        # Optimization loop
        for _ in range(num_steps):
            optimizer.step(closure)
        
        # Save result
        output_path = "output_styled_image.jpg"
        self.save_image(input_img, output_path)
        print(f"\nStyle transfer complete! Result saved as '{output_path}'")
        return output_path

def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Initialize style transfer
    transfer = StyleTransfer()
    
    # Get image paths from user
    content_path = input("Enter path to content image: ")
    style_path = input("Enter path to style image: ")
    
    # Get parameters
    num_steps = int(input("Enter number of steps (default 300): ") or "300")
    content_weight = float(input("Enter content weight (default 1): ") or "1")
    style_weight = float(input("Enter style weight (default 1e6): ") or "1000000")
    
    # Perform style transfer
    output_path = transfer.transfer_style(
        content_path=content_path,
        style_path=style_path,
        num_steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight
    )
    
    # Display result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(content_path))
    plt.title("Content Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(Image.open(style_path))
    plt.title("Style Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(Image.open(output_path))
    plt.title("Generated Image")
    plt.axis("off")
    
    plt.show()

if __name__ == "__main__":
    main()
