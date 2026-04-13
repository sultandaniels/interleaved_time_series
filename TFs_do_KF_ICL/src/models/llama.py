from transformers import LlamaModel, LlamaConfig
from models.lightning_base_model import BaseModel
from core import Config
import torch
import torch.nn as nn

config = Config()

class Llama(BaseModel):
    def __init__(self, n_dims_in, n_embd, n_interm_embd, n_layer=12, n_head=8, n_dims_out=5, learning_rate=config.learning_rate):
        super().__init__(learning_rate)

        # Store these as instance attributes for later access
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out

        llama_configuration = LlamaConfig(
            vocab_size=1, hidden_size=n_embd, intermediate_size=n_interm_embd, num_hidden_layers=n_layer, num_attention_heads=n_head,num_key_value_heads=1, torch_dtype=torch.bfloat16
        )

        llama_configuration._attn_implementation = "flash_attention_2"  # Use flash attention if available

        self._read_in = nn.Linear(n_dims_in, n_embd)
        self._backbone = LlamaModel(llama_configuration)
        del self._backbone.embed_tokens

        self._read_out = nn.Linear(n_embd, n_dims_out)

        self.name = f"llama_embd={n_embd}_interm_embd={n_interm_embd}_layer={n_layer}_head={n_head}"


    def predict_step(self, input_dict, batch_idx=None):
        current = input_dict["current"]
        embeds = self._read_in(current)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        # predict only on xs
        return input_dict, {"preds": prediction}

    def forward(self, input_dict, batch_idx=None, return_intermediate_dict=False):
        input_dict, intermediate_dict = self.predict_step(input_dict, batch_idx)

        # Calculate all loss terms and metrics (scores)
        output_dict = self.calculate_losses_and_metrics(input_dict, intermediate_dict)

        # Calculate optimized loss
        optimized_loss = 0
        for key, loss in output_dict.items():
            if "loss_" in key:
                optimized_loss += loss
        output_dict["optimized_loss"] = optimized_loss
        return (intermediate_dict, output_dict) if return_intermediate_dict else output_dict

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict):
        
        # Calculate loss
        ys = input_dict["target"]

        preds = intermediate_dict["preds"]
        res_sq = (preds - ys) ** 2 #residuals squared

        if config.multi_sys_trace:

            if config.mem_suppress and config.masking:
                #create a mask to identify rows of ys that are all zeros and also all indices from the list of lists input_dict["mask_idx"]
                #ys is of shape [batch_size, seq_len, dims]
                mask_all_zeros = torch.all(ys == 0, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]

                mask_selected_indices = torch.zeros_like(mask_all_zeros, dtype=torch.bool)
                for b, idx_list in enumerate(input_dict["mask_idx"]):
                    mask_idx_minus_1 = [int(idx) - 1 for idx in idx_list]
                    mask_selected_indices[b, mask_idx_minus_1, :] = True #since the target has one less entry than the full segment we need to subtract 1 from the index

                # Log loss on masked indices before zeroing them out
                if mask_selected_indices.any():
                    masked_res = res_sq[mask_selected_indices.expand_as(res_sq)]
                    loss_masked_indices = masked_res.mean()
                else:
                    loss_masked_indices = torch.tensor(0.0, device=res_sq.device)

                mask = mask_all_zeros | mask_selected_indices #combine the two masks with a logical OR

            else:
                loss_masked_indices = None
                # Create a mask to identify rows of ys that are all zeros
                mask = torch.all(ys == 0, dim=-1, keepdim=True)

            # Apply the mask to res_sq to disregard the residuals for rows of ys that are all zeros
            res_sq = res_sq.masked_fill(mask, 0)

            output_dict = {"loss_mse": torch.sum(res_sq) / (~mask).sum()} #mean squared error loss
            if loss_masked_indices is not None:
                output_dict["metric_masked_indices_mse"] = loss_masked_indices
        else:
            output_dict = {"loss_mse": torch.mean(res_sq)}

        # Per-timestep MSE (averaged across batch and dims)
        for i in range(ys.shape[1]):
            output_dict[f"metric_mse_timestep_{i}"] = torch.mean(res_sq[:, i, :])

        return output_dict

    def predict_ar(self, ins, fix_window_len=True):
        ins = torch.from_numpy(ins).float().to(self.device)
        one_d = False
        if ins.ndim == 2:
            one_d = True
            ins = ins.unsqueeze(0)
        bsize, points, _ = ins.shape
        d_o = self.n_dims_out
        outs = torch.zeros(bsize, 1, d_o).to(self.device)
        with torch.no_grad():
            for i in range(1, points + 1):
                I = ins[:, :i]
                if fix_window_len and I.shape[1] > self.n_positions:
                    I = I[:, -self.n_positions:]
                _, interm = self.predict_step({"xs": I})
                pred = interm["preds"][:, -1:]  # b, 1, d_o
                outs = torch.cat([outs, pred], dim=1)
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs
    