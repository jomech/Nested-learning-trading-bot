import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. The Architecture (Deep Nested Logic) ---
'''class DeepMemoryHead(nn.Module):
    def __init__(self, hidden_dim, decay_rate):
        super().__init__()
        self.gamma = decay_rate
        self.hidden_dim = hidden_dim
        # Deep MLP for complex market logic
        self.optimizer_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

    def forward(self, x_input, memory_matrix):
        pred = torch.matmul(memory_matrix, x_input)
        error = (pred - x_input).detach()
        context = torch.cat([error, x_input], dim=-1)
        
        raw_update = self.optimizer_mlp(context)
        update_matrix = raw_update.view(self.hidden_dim, self.hidden_dim)
        update_proposal = torch.tanh(update_matrix)
        
        new_memory = (self.gamma * memory_matrix) + \
                     ((1 - self.gamma) * update_proposal)
        return pred, new_memory'''
        
class DeepMemoryHead(nn.Module):
    def __init__(self, hidden_dim, initial_decay_bias=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. The Logic Unit (Same as before)
        self.optimizer_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        # 2. NEW: The "Forget Gate" (The Panic Button)
        # Input: Context (Error + Data) -> Output: A number between 0 and 1
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Forces output to be between 0% and 100%
        )

    def forward(self, x_input, memory_matrix):
        # A. Make Prediction
        pred = torch.matmul(memory_matrix, x_input)
        error = (pred - x_input).detach()
        context = torch.cat([error, x_input], dim=-1)
        
        # B. Calculate Update Proposal
        raw_update = self.optimizer_mlp(context)
        update_matrix = raw_update.view(self.hidden_dim, self.hidden_dim)
        update_proposal = torch.tanh(update_matrix)
        
        # C. NEW: Calculate Dynamic Gamma (How much to remember?)
        # If the error is HUGE, the gate should drop to 0.0 (Forget)
        forget_gate = self.gate_layer(context) 
        
        # D. Update Memory with Dynamic Gate
        # Memory = (Gate * Old) + ((1 - Gate) * New)
        new_memory = (forget_gate * memory_matrix) + \
                     ((1 - forget_gate) * update_proposal)
        return pred, new_memory

'''class MultiScaleHOPE(nn.Module):
    def __init__(self, hidden_dim, gammas):
        super().__init__()
        self.heads = nn.ModuleList([DeepMemoryHead(hidden_dim, g) for g in gammas])
        self.output_projection = nn.Linear(hidden_dim * len(gammas), hidden_dim)

    def forward(self, x_input, memory_states):
        head_outputs = []
        new_memory_states = []
        for i, head in enumerate(self.heads):
            pred, new_mem = head(x_input, memory_states[i])
            head_outputs.append(pred)
            new_memory_states.append(new_mem)
        
        combined_view = torch.cat(head_outputs, dim=-1)
        final_prediction = self.output_projection(combined_view) + x_input
        return final_prediction, new_memory_states

    def init_memory(self):
        return [torch.zeros(16, 16) for _ in self.heads]'''
        
# --- REPLACEMENT CLASS ---
class AttentionHOPE(nn.Module):
    def __init__(self, hidden_dim, num_heads): # removed 'gammas'
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Just create 'num_heads' identical heads. 
        # They will naturally DIVERGE and learn different decay rates.
        self.heads = nn.ModuleList([DeepMemoryHead(hidden_dim) for _ in range(num_heads)])
        
        # ... (rest of AttentionHOPE is exactly the same)
        
        # The Conductor (Attention Mechanism)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads), 
            nn.Softmax(dim=0) 
        )
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) 

    def forward(self, x_input, memory_states):
        head_outputs = []
        new_memory_states = []
        
        # 1. Run all heads
        for i, head in enumerate(self.heads):
            pred, new_mem = head(x_input, memory_states[i])
            head_outputs.append(pred)
            new_memory_states.append(new_mem)
            
        # 2. The Conductor decides trust
        trust_scores = self.attention_net(x_input) 
        
        # 3. Weighted Combination
        combined_view = torch.zeros_like(x_input)
        for i, output in enumerate(head_outputs):
            combined_view += trust_scores[i] * output
            
        # 4. Final Projection + Residual + Norm
        out = self.output_projection(combined_view) + x_input
        final_prediction = self.layer_norm(out)
        
        return final_prediction, new_memory_states
    
    def init_memory(self):
        # Now this will work because self.hidden_dim exists!
        return [torch.zeros(self.hidden_dim, self.hidden_dim) for _ in self.heads]
class DeepMarketHOPE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Market Gammas:
        # 0.0 (Day Trade), 0.5 (Weekly), 0.9 (Quarterly), 0.99 (Yearly Trend)
        self.layers = nn.ModuleList([
            #MultiScaleHOPE(hidden_dim, [0.0, 0.5, 0.9, 0.99])
            #AttentionHOPE(hidden_dim, [0.0, 0.5, 0.9, 0.99])
            # Pass '4' (number of heads) instead of the list of gammas
            AttentionHOPE(hidden_dim, num_heads=4) 
            for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_vec, all_layer_memories):
        x = torch.tanh(self.encoder(input_vec))
        new_all_memories = []
        for i, layer in enumerate(self.layers):
            x, new_layer_mems = layer(x, all_layer_memories[i])
            new_all_memories.append(new_layer_mems)
        prediction = self.decoder(x)
        return prediction, new_all_memories

    def init_memories(self):
        return [layer.init_memory() for layer in self.layers]

# --- 2. Data Loading & Improved Preprocessing ---
print("Downloading S&P 500 Data (2019-2021)...")
data = yf.download("SPY", start="2019-01-01", end="2021-01-01", progress=False)

# Feature Engineering
data['Return'] = data['Close'].pct_change()
#data['Volatility'] = data['Return'].rolling(window=5).std()
# Shift by 1 to ensure we only see PAST volatility
data['Volatility'] = data['Return'].rolling(window=5).std().shift(1)
data = data.dropna()

# --- IMPROVED NORMALIZATION (ROLLING WINDOW) ---
# Calculate rolling stats (60-day window) to mimic a real trader's view
# Shift by 1 so we don't include today's value in the mean (prevent leakage)
rolling_mean_ret = data['Return'].rolling(window=60).mean().shift(1)
rolling_std_ret = data['Return'].rolling(window=60).std().shift(1)

rolling_mean_vol = data['Volatility'].rolling(window=60).mean().shift(1)
rolling_std_vol = data['Volatility'].rolling(window=60).std().shift(1)
# Add Volume Stats
rolling_mean_volume = data['Volume'].rolling(window=60).mean().shift(1)
rolling_std_volume = data['Volume'].rolling(window=60).std().shift(1)

def get_market_vector(row_idx):
    # 1. Helper to clean single-item Series (MOVED TO TOP)
    # This prevents the "Ambiguous truth value" error
    def clean(val):
        return val.item() if isinstance(val, pd.Series) else val

    # 2. Fetch Rolling Stats (and clean them immediately)
    rm_r = clean(rolling_mean_ret.iloc[row_idx])
    rs_r = clean(rolling_std_ret.iloc[row_idx])
    
    rm_v = clean(rolling_mean_vol.iloc[row_idx])
    rs_v = clean(rolling_std_vol.iloc[row_idx])
    
    rm_vol = clean(rolling_mean_volume.iloc[row_idx])
    rs_vol = clean(rolling_std_volume.iloc[row_idx])

    # 3. Fallback: If any history is missing, return zeros
    # Now rm_r, etc. are simple floats, so pd.isna() works perfectly
    if pd.isna(rm_r) or pd.isna(rs_r) or pd.isna(rm_vol):
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
    # 4. Get Raw Data for Today
    raw_r = clean(data['Return'].iloc[row_idx])
    raw_v = clean(data['Volatility'].iloc[row_idx])
    raw_vol = clean(data['Volume'].iloc[row_idx])

    # 5. Normalize All 3 Inputs
    norm_r = (raw_r - rm_r) / (rs_r * 2)
    norm_v = (raw_v - rm_v) / (rs_v * 2)
    norm_vol = (raw_vol - rm_vol) / (rs_vol * 2)

    # 6. Return Tensor of Size 3
    return torch.tensor([float(norm_r), float(norm_v), float(norm_vol)], dtype=torch.float32)

# --- 3. The Simulation ---
torch.manual_seed(42)
hidden_dim = 16

model = DeepMarketHOPE(input_dim=3, hidden_dim=hidden_dim, num_layers=2)
meta_optimizer = optim.Adam(model.parameters(), lr=0.001) 
all_memories = model.init_memories()

print(f"{'Date':<12} | {'True Ret':<10} | {'Pred Ret':<10} | {'Loss':<8} | {'Status'}")
print("-" * 65)

# Start from 60 because rolling window needs 60 days of history
true_values = []
pred_values = []
dates = []
for i in range(60, len(data)):
    
    # --- FORECASTING MODE ---
    # Input: What we knew YESTERDAY (i-1)
    # Target: What happened TODAY (i)
    input_vector = get_market_vector(i-1) 
    target_vector = get_market_vector(i)

    # If input is empty/zero (start of data), skip
    if torch.all(input_vector == 0):
        continue
    
    # --- Forward Pass (Predict Today based on Yesterday) ---
    pred_vec, all_memories = model(input_vector, all_memories)
    
    # --- Learning ---
    loss = torch.mean((pred_vec - target_vector)**2)
    
    meta_optimizer.zero_grad()
    loss.backward()
    meta_optimizer.step()
    
    # Detach memories
    all_memories = [[m.detach() for m in layer_mems] for layer_mems in all_memories]
    
    # --- Reporting ---
    # Extract raw data for printing
    row = data.iloc[i]
    date_str = str(row.name.date())
    true_ret_val = row['Return']
    
    # Handle Series vs Float for printing
    if isinstance(true_ret_val, pd.Series):
        true_ret_val = true_ret_val.item()
    
    true_ret_pct = true_ret_val * 100
    
    # De-normalize prediction for display
    # Pred = (Norm * Std * 2) + Mean (using yesterday's stats)
    # We use stats from i-1 because that's the "ruler" the model used
    rs_r = rolling_std_ret.iloc[i-1]
    rm_r = rolling_mean_ret.iloc[i-1]
    
    if isinstance(rs_r, pd.Series): rs_r = rs_r.item()
    if isinstance(rm_r, pd.Series): rm_r = rm_r.item()
    
    pred_ret_val = (pred_vec[0].item() * (rs_r * 2)) + rm_r
    pred_ret_pct = pred_ret_val * 100
    
    true_values.append(true_ret_pct)
    pred_values.append(pred_ret_pct)
    dates.append(date_str)
    
    # --- NEW STATUS LOGIC ---
    # We still calculate is_crash to decide WHEN to print (big market moves)
    is_crash = (abs(true_ret_pct) > 3.0)
    
    # But we define STATUS based on the Model's Confusion (Loss)
    loss_val = loss.item()

    if loss_val > 1.7:
        status = "🚨 PANIC"       # Model is broken/surprised
    elif loss_val > 0.5:
        status = "⚠️ Volatile"    # Model is struggling but trying
    else:
        status = "✅ Stable"      # Model is comfortable

    # Print if it's a big move OR every 20 days OR if the model is panicking
    if is_crash or loss_val > 2.0 or i % 20 == 0:
        print(f"{date_str:<12} | {true_ret_pct:>6.2f}%    | {pred_ret_pct:>6.2f}%    | {loss.item():.4f}   | {status}")
        


# ... (End of your loop) ...

# --- PLOTTING ---
# We need to collect lists during the loop to plot them
# (You would need to add: true_rets = [], pred_rets = [] inside the loop)

print("\nGenerating 'crash_recovery.png'...")

# Fake data generation for the example snippet (In your code, append inside the loop!)
# dates = [list of dates]
# true_values = [list of true_ret_pct]
# pred_values = [list of pred_ret_pct]

plt.figure(figsize=(12, 6))
plt.plot(true_values, label='Actual Market Move', color='grey', alpha=0.5)
plt.plot(pred_values, label='AI Prediction', color='red', linewidth=2)
plt.title("Neural Optimizer: Adapting to the COVID-19 Crash")
plt.legend()
plt.savefig("crash_recovery.png")
print("Graph saved!")