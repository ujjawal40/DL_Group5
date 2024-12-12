import matplotlib.pyplot as plt


def plot_batch(imgs, msks=None, size=5):

    # Ensure size does not exceed batch size
    size = min(size, len(imgs))

    # Set up the figure
    plt.figure(figsize=(size * 5, 10))
    for idx in range(size):
        # Plot the image
        plt.subplot(2, size, idx + 1)
        img = imgs[idx].permute((1, 2, 0)).cpu().numpy()  # Convert to HWC format
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] range
        img = img.squeeze()  # Remove the channel dimension for grayscale
        plt.imshow(img, cmap="gray")  # Use gray colormap for better visualization
        plt.axis("off")
        plt.title("Image")

        # Plot the mask if available
        if msks is not None:
            plt.subplot(2, size, idx + 1 + size)
            mask = msks[idx].permute((1, 2, 0)).cpu().numpy()  # Convert to HWC format
            mask_combined = mask.sum(axis=-1)  # Combine classes for better visualization
            plt.imshow(mask_combined, cmap="viridis", alpha=0.8)
            plt.axis("off")
            plt.title("Mask")

    plt.tight_layout()
    plt.show()
