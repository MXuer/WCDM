import torch
import torch.nn as nn

class SignalCNN(nn.Module):
    def __init__(self, input_channels=4, output_dim=160):
        super(SignalCNN, self).__init__()
        # Initial 2D convolution to process the (H, W) = (10240, 3) part 
        # for each of the input_channels.
        # Input shape: (N, C_in, H, W) = (N, 4, 10240, 3)
        self.conv2d_initial = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
            # Output: (N, 16, 10240, 1)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # After this, we squeeze the last dimension to get (N, 16, 10240) for 1D convolutions.

        # 1D Convolutional blocks
        self.conv1d_block1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3), # padding='same' essentially before pooling
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # Output length: 10240 / 4 = 2560
        )
        self.conv1d_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # Output length: 2560 / 4 = 640
        )
        self.conv1d_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # Output length: 640 / 4 = 160
        )
        self.conv1d_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # Output length: 160 / 4 = 40
        )
        self.conv1d_block5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # Output length: 40 / 4 = 10
        )

        # Flatten and Fully Connected layers
        # Output from conv1d_block5 is (N, 512, 10)
        # Flattened size: 512 * 10 = 5120
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.Sigmoid() # Sigmoid for 0/1 output, suitable for BCELoss
        )

    def forward(self, x):
        # x initial shape: (N, 4, 10240, 3)
        x = self.conv2d_initial(x) # Output: (N, 16, 10240, 1)
        x = x.squeeze(-1) # Output: (N, 16, 10240), ready for 1D conv

        x = self.conv1d_block1(x) # Output: (N, 32, 2560)
        x = self.conv1d_block2(x) # Output: (N, 64, 640)
        x = self.conv1d_block3(x) # Output: (N, 128, 160)
        x = self.conv1d_block4(x) # Output: (N, 256, 40)
        x = self.conv1d_block5(x) # Output: (N, 512, 10)
        
        x = self.fc_block(x) # Output: (N, 160)
        return x

# Example usage (for testing the model structure):
if __name__ == '__main__':
    # Create a dummy input tensor with the specified dimensions
    # batch_size = 10, channels = 4, height = 10240, width = 3
    test_input = torch.randn(10, 4, 10240, 3)
    
    # Instantiate the model
    model = SignalCNN(input_channels=4, output_dim=160)
    
    # Pass the input through the model
    output = model(test_input)
    
    # Print the output shape to verify
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Expected: torch.Size([10, 160])

    # Check if output values are between 0 and 1 (due to Sigmoid)
    print(f"Min output value: {output.min().item()}")
    print(f"Max output value: {output.max().item()}")