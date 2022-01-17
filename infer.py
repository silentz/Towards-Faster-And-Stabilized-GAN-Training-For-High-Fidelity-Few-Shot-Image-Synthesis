import argparse
import torch
import torchvision

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to traced model')
    parser.add_argument('--n_samples', type=int, default=4, help='num samples to generate')
    parser.add_argument('--output', type=str, default='samples.png', help='output image path')
    args = parser.parse_args()

    model = torch.jit.load(args.model)
    input = torch.randn(args.n_samples, 256, 1, 1)

    output = model(input)
    torchvision.utils.save_image(output, args.output)
