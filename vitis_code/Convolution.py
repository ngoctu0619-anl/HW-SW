class TinyImageCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3) # 1 -> 8 kênh
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3) # 8 -> 16 kênh
        self.fc = nn.Linear(16 * 14 * 14, num_classes) # FC layer

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = TinyImageCNN(num_classes).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda'))
        loss = criterion(outputs, labels.to('cuda'))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} hoàn thành.")
