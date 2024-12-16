import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
num_epochs = 5 #########
# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def amplify_bias_deltas(bias_deltas, amplification_factor):
    return np.exp(amplification_factor * bias_deltas) - 1

# Load the raw attack vectors and labels
logger.info("Loading raw attack vectors and labels...")
attack_vectors = np.load(f"attack_vectors{num_epochs}.npy")
attack_labels = np.load(f"attack_labels{num_epochs}.npy")

# Log original statistics
logger.info(f"Original shape: {attack_vectors.shape}")
logger.info(f"Original range: [{attack_vectors.min():.6f}, {attack_vectors.max():.6f}]")

# Try different amplification factors
amplification_factors = [1, 50, 100, 200, 300, 500]
plt.figure(figsize=(20, 12))

for i, factor in enumerate(amplification_factors):
    # Compute magnitudes and amplify
    amplified_vectors = amplify_bias_deltas(attack_vectors, factor)
    magnitudes = np.mean(np.abs(amplified_vectors), axis=1)
    
    # Create subplot
    plt.subplot(2, 3, i+1)
    sns.histplot(data={
        'Member': magnitudes[attack_labels == 1],
        'Non-member': magnitudes[attack_labels == 0]
    }, 
    common_norm=True,
    stat='density',
    alpha=0.6)
    
    plt.title(f'Exponential Amplification Factor: {factor}')
    plt.xlabel('Amplified Mean Magnitude')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Log statistics
    member_mags = magnitudes[attack_labels == 1]
    non_member_mags = magnitudes[attack_labels == 0]
    logger.info(f"\nAmplification factor: {factor}")
    logger.info(f"Member mean: {np.mean(member_mags):.6f} ± {np.std(member_mags):.6f}")
    logger.info(f"Non-member mean: {np.mean(non_member_mags):.6f} ± {np.std(non_member_mags):.6f}")
    logger.info(f"Range: [{np.min(magnitudes):.6f}, {np.max(magnitudes):.6f}]")

# Save amplified vectors with factor 500
amplified_vectors_200 = amplify_bias_deltas(attack_vectors, 200)
np.save(f'amplified_vectors{num_epochs}.npy', amplified_vectors_200)
np.save(f'attack_labels{num_epochs}.npy', attack_labels)
logger.info(f"Amplified vectors and labels with factor 500 saved to 'amplified_vectors{num_epochs}.npy' and 'amplified_labels{num_epochs}.npy'")

plt.tight_layout()
plt.savefig(f'exponential_attack_vector_amplifications{num_epochs}.png')
plt.close()

logger.info(f"Amplification distributions plot saved as 'exponential_attack_vector_amplifications{num_epochs}.png'")
