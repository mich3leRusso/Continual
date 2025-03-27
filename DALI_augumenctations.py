import nvidia.dali as dali
from nvidia.dali import fn
import torch 
from parse import args
import cupy as cp
class DALITransformPipeline(dali.pipeline.Pipeline):
    def __init__(self, batch_size, device_id=0):
        # Initialize the pipeline with the given batch size, thread count, and device id.
        super(DALITransformPipeline, self).__init__(batch_size, num_threads=2, device_id=device_id)
        
        # Define an external source with an explicit name ("input").
        self.input = fn.external_source(device="cpu", name="input")
        
        # Apply a random resized crop to the input images.
        self.cropped = fn.random_resized_crop(self.input,
                                              size=(64, 64),
                                              random_area=[0.1, 1.0],
                                              random_aspect_ratio=[0.5, 2.0])
        # Apply color jitter (color twist) to the cropped images.
        self.jittered = fn.color_twist(self.cropped, brightness=0.5, contrast=0.5)
        
        # Use crop_mirror_normalize for normalization.
        # Note: crop_mirror_normalize expects images in HWC order.
        # Here, we assume the images are in the [0, 255] range.
        # The mean and std values are scaled by 255 to match that range.
        self.normalized = fn.crop_mirror_normalize(
            self.jittered,
            crop=(64, 64),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout="CHW"  # output as (C, H, W)
        )

    def define_graph(self):
        # Build the graph: input -> crop -> color jitter -> normalization.
        return self.normalized

def apply_transforms_and_permute_dali(batch, num_permutations, device_id=0):
    # Get the batch size from the PyTorch batch.
    batch_size = batch.shape[0]

    # Convert the batch from PyTorch's NCHW to NHWC order for DALI.
    images_gpu = batch.to("cuda").permute(0, 2, 3, 1).contiguous()

    # Create and build the DALI pipeline.
    pipe = DALITransformPipeline(batch_size=batch_size, device_id=device_id)
    pipe.build()

    # Feed the input images into the pipeline.
    # The name here ("input") must match the one specified in fn.external_source.
    pipe.feed_input("input", images_gpu)

    try:
        # Execute the pipeline.
        pipe_out = pipe.run()
        print(f"Pipeline output shape: {pipe_out[0].shape}")
    except RuntimeError as e:
        print(f"Error during pipeline execution: {e}")
        return None

    # Convert the output DALI tensor to a PyTorch tensor.
    # Since we set output_layout="CHW", the output is already in NCHW order.
    augmented_batch = pipe_out[0].as_tensor()
    augmented_batch= cp.asarray(augmented_batch)
    augmented_batch = torch.tensor(augmented_batch, device=args.device)

    return augmented_batch

# Example usage:
# Assuming `batch` is a PyTorch tensor of shape (N, C, H, W) with pixel values in [0, 255].
# final_output = apply_transforms_and_permute_dali(batch, num_permutations=5)
