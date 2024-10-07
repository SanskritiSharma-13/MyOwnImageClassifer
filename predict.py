import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import json

def load_checkpoint(filepath):
    """Load a checkpoint and rebuild the model."""
    checkpoint = torch.load(filepath)

    # Load the appropriate model based on the type in the checkpoint
    model_map = {
        "vgg11": torchvision.models.vgg11(pretrained=True),
        "vgg13": torchvision.models.vgg13(pretrained=True),
        "vgg16": torchvision.models.vgg16(pretrained=True),
        "vgg19": torchvision.models.vgg19(pretrained=True)
    }
    
    model = model_map.get(checkpoint['vgg_type'], torchvision.models.vgg16(pretrained=True))
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """Process an image for use in a PyTorch model."""
    pil_image = Image.open(image_path).convert("RGB")

    # Resize and crop the image
    pil_image = pil_image.resize((256, 256))
    
    # Center crop
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2

    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert color channels from 0-255 to 0-1
    np_image = np.array(pil_image) / 255.0

    # Normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose color channel to 1st dimension
    np_image = np_image.transpose((2, 0, 1))

    # Convert to FloatTensor
    tensor = torch.from_numpy(np_image).type(torch.FloatTensor)

    return tensor

def predict(image_path, model, topk, device, cat_to_name):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))
        
    top_ps, top_classes = ps.topk(topk, dim=1)
    
    idx_to_flower = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes[0].tolist()]

    return top_ps[0].tolist(), predicted_flowers_list

def print_predictions(args):
    """Load model and predict the class of the input image."""
    model = load_checkpoint(args.model_filepath)

    # Decide device depending on user arguments and device availability
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if args.gpu and not torch.cuda.is_available():
        print("GPU was selected, but no GPU is available. Using CPU instead.")

    model = model.to(device)

    # Load category names
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # Predict the image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i + 1, top_classes[i], top_ps[i] * 100))

if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('image_filepath', help="Path to the image file you want to classify")
    parser.add_argument('model_filepath', help="File path of the checkpoint file, including the extension")

    # Optional arguments
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath',
                        help="Path to a JSON file that maps categories to real names", 
                        default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', 
                        help="Number of most likely classes to return, default is 5", 
                        default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', 
                        help="Include this argument to use the GPU for inference", 
                        action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)
