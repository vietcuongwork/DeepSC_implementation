import torch
from torch.utils.data import DataLoader
import json
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import create_masks, power_normalize, subsequent_mask


# Mock Channels class (simplified for shape inspection)
class Channels:
    def Rayleigh(self, Tx_sig, n_var):
        return Tx_sig, 0

    def AWGN(self, Tx_sig, n_var):
        return Tx_sig, 0

    def Rician(self, Tx_sig, n_var):
        return Tx_sig, 0


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load vocabulary
with open('data/vocab.json', 'rb') as f:
    vocab = json.load(f)
token_to_idx = vocab['token_to_idx']
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]

# Define MAX_LENGTH
MAX_LENGTH = 30

# Load test dataset
test_eur = EurDataset('test')

# Find a sample that requires padding (shorter than MAX_LENGTH)
for sample in test_eur:
    if len(sample) < MAX_LENGTH:
        sents_list = sample
        break
else:
    raise ValueError("No sample found that is shorter than MAX_LENGTH")

print("Selected sample length before padding:", len(sents_list))


# Define model arguments
class Args:
    def __init__(self):
        self.num_layers = 4
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.batch_size = 1  # Single sample for debugging


args = Args()

# Create a DataLoader with the selected sample
batch = [sents_list]
test_iterator = DataLoader(batch, batch_size=args.batch_size, num_workers=0,
                           pin_memory=True, collate_fn=collate_data)

# Get the padded batch
for sents in test_iterator:
    sents = sents.to(device)
    break  # Take only the first batch (single sample)

print("Padded batch shape:", sents.shape)

# Initialize model
num_vocab = len(token_to_idx)
deepsc = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                args.d_model, args.num_heads, args.dff, 0.1).to(device)
deepsc.eval()

# Set channel and noise variance
channel = 'Rayleigh'
n_var = 0.1


# Function to inspect shapes during training step
def inspect_train_step_rayleigh(model, src, trg, n_var, pad, channel):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    print("Input sequence indices:", src[0].tolist())
    print("Target sequence indices:", trg_real[0].tolist())
    print("trg_inp shape:", trg_inp.shape)
    print("trg_real shape:", trg_real.shape)

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    print("Input source shape:", src.shape)

    # Semantic Encoder
    enc_output = model.encoder(src, src_mask)
    print("After semantic encoder:")
    print("  Shape:", enc_output.shape)
    print("  Sample values:", enc_output[0, 0, :5].tolist())

    # Channel Encoder
    channel_enc_output = model.channel_encoder(enc_output)
    print("After channel encoder:")
    print("  Shape:", channel_enc_output.shape)
    print("  Sample values:", channel_enc_output[0, 0, :5].tolist())

    # Power Normalization
    Tx_sig = power_normalize(channel_enc_output)
    print("After power normalization:")
    print("  Shape:", Tx_sig.shape)
    print("  Sample values:", Tx_sig[0, 0, :5].tolist())

    # Channel
    channels = Channels()
    if channel == 'Rayleigh':
        Rx_sig, snr = channels.Rayleigh(Tx_sig, n_var)
    else:
        Rx_sig = Tx_sig
    print("After Rayleigh channel:")
    print("  Shape:", Rx_sig.shape)
    print("  Sample values:", Rx_sig[0, 0, :5].tolist())

    # Channel Decoder
    channel_dec_output = model.channel_decoder(Rx_sig)
    print("After channel decoder:")
    print("  Shape:", channel_dec_output.shape)
    print("  Sample values:", channel_dec_output[0, 0, :5].tolist())

    # Semantic Decoder
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask,
                               src_mask)
    print("After semantic decoder:")
    print("  Shape:", dec_output.shape)
    print("  Sample values:", dec_output[0, 0, :5].tolist())

    # Dense Layer
    pred = model.dense(dec_output)
    print("After dense layer:")
    print("  Shape:", pred.shape)
    print("  Sample values (first 5 logits):", pred[0, 0, :5].tolist())

    # Predicted sequence indices
    predicted_indices = torch.argmax(pred, dim=-1)[0].tolist()
    print("Predicted sequence indices:", predicted_indices)


# Greedy decode function
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol,
                  channel):
    """
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(
        device)  # [batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = power_normalize(
        channel_enc_output)  # Corrected from PowerNormalize to power_normalize

    if channel == 'AWGN':
        Rx_sig, _ = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig, _ = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig, _ = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(
        src.data).to(device)

    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(
            torch.FloatTensor).to(device)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(
            torch.FloatTensor).to(device)
        combined_mask = torch.max(trg_mask, look_ahead_mask)

        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)

        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs


# Run the inspection
# print("Running training step inspection...")
# inspect_train_step_rayleigh(deepsc, sents, sents, n_var, pad_idx, channel)

# Run greedy decode
print("\nRunning greedy decode...")
decoded_output = greedy_decode(deepsc, sents, n_var, MAX_LENGTH, pad_idx,
                               start_idx, channel)
print("Decoded sequence indices:", decoded_output[0].tolist())
