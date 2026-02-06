"""
==============================================================================
CM-DLSSM Artifact: Baseline II - CodeBERT with Truncation (Weak Baseline)
Path: src/baselines/codebert_limit.py
==============================================================================
Reference: Section 1.2 (Limitations of SOTA) - "Truncation Crisis"
           Section 6.1 (Benchmarks) - Performance Gap
           Section 6.2 (Efficiency) - "Throughput Comparison"

Description:
    This module implements the industry-standard approach to handling long code:
    TRUNCATION.

    It wraps a standard `microsoft/codebert-base` model. If the input sequence
    is longer than `max_position_embeddings` (usually 512), it blindly cuts
    off the tail.

    Scientific Purpose:
    1. To demonstrate the "Blind Spot": If a vulnerability relies on a Sink
       at token index 5000, this model returns a random guess (or safe),
       resulting in low Recall for "Chained Vulnerabilities".
    2. To benchmark Efficiency: Used as the "Short Context" speed baseline
       in Figure 7.
==============================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class TruncatedCodeBERT(nn.Module):
    def __init__(
        self, 
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        force_truncation_limit: int = 512
    ):
        """
        Args:
            model_name: Pretrained HF model path.
            force_truncation_limit: The hard limit (usually 512 for BERT).
                                    Any token beyond this is discarded.
        """
        super().__init__()
        self.truncation_limit = force_truncation_limit
        
        print(f"[Baseline] Loading Truncated CodeBERT ({model_name})...")
        print(f"[Baseline] HARD LIMIT: First {self.truncation_limit} tokens ONLY.")
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.hidden_dropout_prob = dropout
        self.config.problem_type = "single_label_classification"
        
        # Load pre-trained model with a classification head
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Args:
            input_ids: (B, L) - L can be huge (e.g., 128k from the dataloader)
            attention_mask: (B, L)
        
        Returns:
            logits: (B, num_labels)
        """
        B, L = input_ids.shape
        
        # 1. THE TRUNCATION STEP (The "Flaw" of this baseline)
        # If L > limit, we slice. If L < limit, we take it all.
        effective_len = min(L, self.truncation_limit)
        
        input_ids_truncated = input_ids[:, :effective_len]
        
        if attention_mask is not None:
            attention_mask_truncated = attention_mask[:, :effective_len]
        else:
            attention_mask_truncated = None

        # 2. Forward pass through the Transformer
        # Note: 'token_type_ids' are usually not needed for CodeBERT in single seq mode
        outputs = self.bert(
            input_ids=input_ids_truncated,
            attention_mask=attention_mask_truncated
        )
        
        return outputs.logits

    def get_token_coverage(self, original_length: int) -> float:
        """
        Helper metric for experiments.
        Returns % of code actually seen by the model.
        """
        if original_length <= self.truncation_limit:
            return 1.0
        return self.truncation_limit / original_length

# ==============================================================================
# Unit Test / Artifact Smoke Test
# ==============================================================================
if __name__ == "__main__":
    print("[*] Testing Truncated CodeBERT Baseline...")
    
    # 1. Setup
    model = TruncatedCodeBERT(force_truncation_limit=512)
    model.eval()
    
    # 2. Simulate a LONG input (e.g., 10,000 tokens)
    # This represents a large file where the vulnerability might be at the end.
    B, L_huge = 2, 10000
    long_input = torch.randint(0, 1000, (B, L_huge))
    
    # 3. Forward
    print(f"Input Shape: {long_input.shape} (10k tokens)")
    logits = model(long_input)
    
    print(f"Output Shape: {logits.shape}")
    
    # 4. Verify Truncation logic happened implicitly inside
    coverage = model.get_token_coverage(L_huge)
    print(f"Effective Coverage: {coverage*100:.2f}%")
    
    # Assertions
    assert logits.shape == (B, 2)
    assert coverage < 0.1, "Model should have seen less than 10% of the code"
    
    print("[+] Test Passed: Model successfully ignored 95% of the data (as expected).")