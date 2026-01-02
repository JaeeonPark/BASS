# BASS

Implementation of BASS (Block-wise Adaptation for Speech Summarization)

## Overview

BASS is a block-wise approach for speech summarization that processes long audio sequences by splitting them into blocks and maintaining semantic context across blocks.

## Key Features

- **Block-wise Processing**: Handles long audio sequences by splitting into manageable blocks
- **Semantic Context**: Maintains semantic information across blocks using a semantic updater
- **Incremental Encoding**: Processes audio blocks sequentially with context carry-over

## Model Architecture

The implementation includes:

- `BlockProcessor`: Splits long audio into fixed-size blocks
- `SemanticUpdater`: Aggregates and updates semantic context across blocks
- `BlockwiseEncoder`: Encodes each audio block (uses pretrained ESPNet ASR encoder)
- `BlockwiseDecoder`: Generates summary text from encoded blocks
- `BASSModel`: Main model orchestrating block-wise processing

## Usage

```python
from bass.model import BASSModel, BASSConfig

# Initialize model with configuration
config = BASSConfig(
    freeze_encoder=True,
    block_size=1000,
    semantic_dim=2048,
    vocab_size=500
)
model = BASSModel(config)

# Process audio blocks
# (See model.py for detailed usage examples)
```

## Requirements

- PyTorch
- ESPNet (for pretrained ASR encoder)
- Other dependencies as specified in the code

## Citation

If you use this implementation, please cite the original BASS paper:

```
@article{sharma2023bass,
  title={Bass: Block-wise adaptation for speech summarization},
  author={Sharma, Roshan and Zheng, Kenneth and Arora, Siddhant and Watanabe, Shinji and Singh, Rita and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2307.08217},
  year={2023}
}
```
