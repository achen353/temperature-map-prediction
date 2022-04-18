import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        num_enc_layers,
        input_size,
        input_len,
        pred_len,
        feature_size,
        NHEAD,
        has_pos=False,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        # num_enc_layers (hyperparameter): number of the encoder layers
        # input_size: number of pixel per map
        # input_len: number of input maps
        # pred_len: number of predict maps
        # feature_size (hyperparameter): feature_size in the transformer encoder
        # NHEAD: number of heads in the transformer encoder
        # has_pos: has position embedding or not
        # dropout: dropout of the transformer encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_len = input_len
        self.pred_len = pred_len
        self.has_pos = has_pos

        self.project_linear = nn.Linear(
            input_size, feature_size
        )  # project input to the feature_size

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=NHEAD, dropout=dropout
        )
        self.transformer_enc = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_enc_layers
        )

        if self.has_pos:
            self.pos_layer = nn.Linear(input_size, input_size)

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

        self.project_linear.bias.data.zero_()
        self.project_linear.weight.data.uniform_(-range, range)

        if self.has_pos:
            self.pos_layer.bias.data.zero_()
            self.pos_layer.weight.data.uniform_(-range, range)

    def forward(self, input, pos_info=None, token_is_zero=True):
        # input: shape is [seq_len, batch_size, input_size]
        # pos_info: shape is [seq_len + pred_len, batch_size, 1]
        # token_is_zero: if True, then use zero tensor, otherwise use the last tensor from the input
        seq_len, batch_size, input_size = input.shape

        if token_is_zero:
            tokens = torch.zeros(
                self.pred_len, batch_size, input_size
            )  # this is for the output seq
        else:
            tokens = input[-1, :, :]
            tokens = tokens.repeat([self.pred_len, 1, 1])

        output = torch.cat((input, tokens.to(self.device)), 0)  # cat two tensors

        if self.has_pos:
            pos_enc = self.pos_layer(
                output
            )  # using linear transform as the position embedding
            output = torch.add(pos_enc, output)  # add position info and input

        output = self.project_linear(output)

        output = self.transformer_enc(output)

        output = output[
            -self.pred_len :, :, :
        ]  # only use the sequence of tokens to predict

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

# model = Transformer(num_enc_layers=3, input_size=400, input_len=20, pred_len=5, feature_size = 512, NHEAD=4, has_pos=True).to(device)
# pos_info = [[1], [2], ..., [25]] # assummed
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
#         pred = model(x, pos_info, token_is_zero = False)
#       ............


###############################################################
