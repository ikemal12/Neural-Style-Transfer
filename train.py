import os
import time
import torch 
import argparse
from neural_style_transfer import (
    run_style_transfer, img_loader, device, cnn,
    cnn_normalization_mean, cnn_normalization_std
)
from utils import save_image, save_comparison, create_output_directory

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', required=True, help='Path to the content image')
    parser.add_argument('--style', required=True, help='Path to the style image')
    parser.add_argument('--output_dir', type=str, default='results/', help='Base output directory')
    parser.add_argument('--imsize', type=int, default=512, help='Size of output image')
    parser.add_argument('--num_steps', type=int, default=300, help='Number of optimization steps')
    parser.add_argument('--style_weight', type=float, default=1000000, help='Weight for style loss')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    parser.add_argument('--init_random', action='store_true', help='Initialize with random noise (default: content image)')
    return parser.parse_args()

def main(args):
    print("Starting Neural Style Transfer...")
    start_time = time.time()
    print(f"Content: {args.content}")
    print(f"Style: {args.style}")
    print(f"Steps: {args.num_steps}")
    print()

    # create output directory
    output_dir = create_output_directory(args.output_dir, args.content, args.style)
    print(f"Output directory created at: {output_dir}\n")

    # load images
    content_img = img_loader(args.content, args.imsize)
    style_img = img_loader(args.style, args.imsize)
    print(f"Content image shape: {content_img.shape}")
    print(f"Style image shape: {style_img.shape}\n")

    # initialize input image
    if args.init_random:
        input_img = torch.randn(content_img.data.size(), device=device)
        print("Initialized with random noise")
    else:
        input_img = content_img.clone()
        print("Initialized with content image")
    print()

    # run style transfer
    print("Starting style transfer...")
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img,
        num_steps=args.num_steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight
    )
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nStyle transfer complete in {minutes} minutes {seconds} seconds\n")

    # save results
    result_path = os.path.join(output_dir, 'result.jpg')
    comparison_path = os.path.join(output_dir, 'comparison.jpg')    
    save_image(output, result_path)
    save_comparison(content_img, style_img, output, comparison_path)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
