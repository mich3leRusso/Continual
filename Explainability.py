import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from parse import args
import shap
from models.gated_resnet32 import GatedConv2d
import torch.nn.functional as F

def lime_tool(input_images, model, labels=0, task_id=0, class2name=[]):


    # Ensure the model is in eval mode
    model.eval()

    # Wrapper to convert input image and output probabilities
    def predict_fn(images,task=0):
        # Convert list of images to tensor
        if images.max() > 1.0:
                 images = images / 255.0  # Normalize to [0, 1]
        images=torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32)
        with torch.no_grad():

            logits = model(images.to(args.device))

            pred = logits[:, task * args.classes_per_exp:(task + 1) * args.classes_per_exp]

            last = logits[:, args.n_classes + task].unsqueeze(1)

            pred = torch.cat([pred, last], dim=1)

            probs = torch.nn.functional.softmax(pred, dim=1)

        return probs.cpu().numpy()

    print(input_images.shape)
    print(input_images[0, :, :, :].shape)
    input()

    for i in range(input_images.shape[0]):
        explainer = lime_image.LimeImageExplainer()
        # Plot


        explanation = explainer.explain_instance(
            image=input_images[i, :, :, :].permute(1,2,0).detach().cpu().numpy(),
            classifier_fn=predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=True
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        image_np = input_images[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        axs[0].imshow(image_np)
        axs[0].axis('off')
        axs[0].set_title("Tensor Image")


        axs[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        axs[1].axis('off')
        axs[1].set_title("LIME Explanation")
        plt.tight_layout()
        plt.show()

    return


#visualizes the most important filters activations
def visualize_meaningfull_layers(model):
    #visualize_network_layers(model)
    conv_index=0
    for m in model.modules():
        if isinstance(m, GatedConv2d):
            #take the first convolutional layer
            conv1=m
            conv_index+=1
            if conv_index>1:
                break #go out the cycle

    conv1_weights = conv1.weight.data.cpu().numpy()

    # The filters are usually of shape (out_channels, in_channels, height, width)
    # To visualize them, we need to focus on the individual filters.

    num_filters = conv1_weights.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))

    for i in range(num_filters):
        # Assuming it's a single-channel input (e.g., grayscale) for simplicity
        # For RGB, you'd need to visualize each channel separately
        axes[i].imshow(conv1_weights[i, 0, :, :], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Filter {i + 1}")

    plt.show()

    return

def visualize_network_layers(model):
    print(model)

    return


def maximal_response(model, channel=0, number_conv=0):

    model.eval()
    # Find the first GatedConv2d layer
    i=0
    for m in model.modules():
        if isinstance(m, GatedConv2d):

            if i==number_conv:
                target_layer = m
                break
            else:
                i+=1
    else:
        raise ValueError("No GatedConv2d layer found.")

    # Hook to capture the activation of the first channel
    activation = {}

    def hook_fn(module, input, output):
        activation["value"] = output[:, channel, :, :]  # channel index 0


    hook = target_layer.register_forward_hook(hook_fn)

    # Initialize input image as noise
    input_image = torch.randn(1, 3, 32, 32, requires_grad=True, device=args.device)

    optimizer = torch.optim.Adam([input_image], lr=0.1)

    for _ in range(5000):
        optimizer.zero_grad()
        model(input_image.to(args.device))
        act = activation.get("value")
        #print(act.shape)
        #print(f"new activation {act.mean()}")
        #input()

        if act is None:
            raise RuntimeError("Activation hook not triggered")

        loss = -act.mean()  # maximize activation
        loss.backward()
        optimizer.step()

    # Visualize the result
    img = input_image
    print(img.shape)
    hook.remove()
    return img


def occlusion_sensitivity(model, input_images, labels_idx=None, patch_size=8, stride=4, device=args.device):
    model.eval()

    for i in range(input_images.shape[0]):
        label_idx=labels_idx[i]
        image=input_images[i,:,:,:]
        heatmap=occlusions_heatmap(model,image, label_idx=label_idx, patch_size=patch_size, stride=stride, device=args.device)

    return heatmap



def occlusions_heatmap(model,image, label_idx=None, patch_size=8, stride=4, device='cuda',task_ids=0):

    image = image.to(device)

    image=image.unsqueeze(dim=0)
    _, C, H, W = image.shape
    true_label=label_idx
    # Get base prediction (before occlusion)
    with torch.no_grad():
        base_output = model(image)

        base_output=base_output[:, (task_ids)*args.classes_per_exp:(task_ids+1)+args.classes_per_exp]

        label_idx = base_output.argmax(dim=1).item()
        base_confidence = base_output[0, label_idx].item()

    # Heatmap to store confidence drop
    heatmap = np.zeros((H // stride, W // stride))

    for i, y in enumerate(range(0, H - patch_size + 1, stride)):
        for j, x in enumerate(range(0, W - patch_size + 1, stride)):
            # Clone and occlude
            occluded = image.clone()
            occluded[:, :, y:y + patch_size, x:x + patch_size] = 0.5  # gray patch

            with torch.no_grad():
                out = model(occluded)
                conf = out[0, label_idx].item()
                drop = base_confidence - conf
                heatmap[i, j] = drop



    # Resize to match input
    heatmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap_resized = F.interpolate(heatmap_t, size=(H, W), mode='bilinear', align_corners=False)
    heatmap_resized = heatmap_resized.squeeze().cpu().numpy()



    # Normalize for visualization
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()

    # Resize heatmap to input size for overlay
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()

    # Convert image to displayable format
    img_np = image.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img_np -= img_np.min()
    img_np /= img_np.max()

# Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_np)
    heat = axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[1].set_title(f'Occlusion Sensitivity (Class {label_idx} and true label {true_label})')
    axes[1].axis('off')

    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04, label='Importance (Confidence Drop)')
    plt.tight_layout()
    plt.show()