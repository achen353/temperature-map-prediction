import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        pred_len,
        feature_size,
        dropout=0,
        bidirectional=False,
    ):
        super(LSTM, self).__init__()
        # num_layers (hyperparameter): number of the encoder layers
        # input_size: number of pixel per map
        # input_len: number of input maps
        # pred_len: number of predict maps
        # feature_size (hyperparameter): feature_size in the transformer encoder
        # NHEAD: number of heads in the transformer encoder
        # has_pos: has position embedding or not
        # dropout: dropout of the transformer encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=feature_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.linear1 = nn.Linear(feature_size, feature_size)
        self.l_relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(feature_size, feature_size)
        self.l_relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(feature_size, input_size)

        self.init_weights()

    def init_weights(self):
        range = 1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-range, range)

        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-range, range)

        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-range, range)

    def forward(self, input):

        # input: shape is [seq_len, batch_size, input_size]
        # pos_info: shape is [seq_len + pred_len, batch_size, 1]
        # token_is_zero: if True, then use zero tensor, otherwise use the last tensor from the input

        seq_len, batch_size, input_size = input.shape

        output, (h_n, c_n) = self.lstm(input, None)

        output = output[-self.pred_len :, :, :]

        output = self.linear1(output)
        output = self.l_relu1(output)
        output = self.linear2(output)
        output = self.l_relu2(output)
        output = self.linear3(output)

        # TODO: Fix output all NaN (@Chen)

        return output


################################################################
################ example to train the model ####################
###############################################################

# model = LSTM(num_layers=3, input_size=400, pred_len=5, feature_size = 512, dropout=0, bidirectional=True).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
# loader = Data.DataLoader(dataset=torch_dataset, batch_size= BATCH_SIZE, shuffle=True)
# model.train()

# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(loader):
#
#         x = x.permute(1,0,2) # make sure batchsize is in the second dim
#
#         y = y.permute(1,0,2) # make sure batch size is in the second dim
#
#         pred = model(x)
#       ............


###############################################################
