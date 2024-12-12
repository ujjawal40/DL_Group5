import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics CSV file
history_df = pd.read_csv('/Users/bhagavathsaidarapureddy/Documents/DL_Project/Testinglocal/metrics.csv')

# Plotting train and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='blue', marker='o')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', color='red', marker='o')
plt.title('Train and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting train and validation Dice score
plt.figure(figsize=(10, 5))
plt.plot(history_df['epoch'], history_df['train_dice'], label='Train Dice Score', color='blue', marker='o')
plt.plot(history_df['epoch'], history_df['val_dice'], label='Validation Dice Score', color='red', marker='o')
plt.title('Train and Validation Dice Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)
plt.show()


