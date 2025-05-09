import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from parse import args
import shap

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

def shap_explainer(model, test_set, class2image=[]):
    return