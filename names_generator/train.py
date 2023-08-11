import torch
import logging
from names_generator import CONFIG
from names_generator.dataset import NamesDataset
from names_generator.model import NamesGeneratorModel


@torch.no_grad()
def estimate_loss(_dataset, _model):
    out = {}
    _model.eval()
    for mode in ['train', 'val']:
        _losses = torch.zeros(CONFIG.eval_interval)
        for k in range(CONFIG.eval_interval):
            _batch_input, _batch_target = _dataset.get_batch(mode, CONFIG.context_length, CONFIG.device, CONFIG.batch_size)
            _logits, _loss = _model(_batch_input, _batch_target)
            _losses[k] = _loss.item()
        out[mode] = _losses.mean()
    _model.train()
    return out


if __name__ == '__main__':
    dataset = NamesDataset(CONFIG.dataset_root_folder)
    model = NamesGeneratorModel(dataset.vocab_size)
    m = model.to(CONFIG.device)
    # print the number of parameters in the model
    logging.info(f"number of params: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)

    for iteration in range(CONFIG.epochs):
        logging.info(f"Epoch {iteration}")
        # every once in a while evaluate the loss on train and val sets
        if iteration % CONFIG.eval_interval == 0 or iteration == CONFIG.epochs - 1:
            losses = estimate_loss(dataset, model)
            logging.info(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        batch_input, batch_target = dataset.get_batch('train', CONFIG.context_length, CONFIG.device, CONFIG.batch_size)

        # evaluate the loss
        logits, loss = model(batch_input, batch_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the model to disk
    torch.save(model.state_dict(), 'resources/model.pth')
