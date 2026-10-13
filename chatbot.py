import torch
import torch.nn.functional as F
from bot import BigramLanguageModel, decode, device, checkpoint_file, block_size, stoi, text

checkpoint = torch.load(checkpoint_file, map_location=device)
model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

MAX_HISTORY_LINES = 4
train_lines = {l.split("> ", 1)[-1].strip() for l in text.split("\n")}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

@torch.no_grad()
def reply(model, history, max_new_tokens=160, temperature=0.5, top_k=8):
    idx = torch.tensor([encode(history + "<friend> ")], dtype=torch.long, device=device)
    out = ""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat((idx, nxt), dim=1)
        out += decode([nxt.item()])
        if "<user_1>" in out:
            break

    hit_limit = "<user_1>" not in out
    lines = out.split("<user_1>")[0].strip().split("\n")
    if hit_limit and len(lines) > 1:
        lines = lines[:-1]
    return "<friend> " + "\n".join(lines)

history = []

while True:
    user_input = input("\n<user_1> ")
    if user_input.lower() == "exit":
        break

    history.append(f"<user_1> {user_input}")
    history = history[-MAX_HISTORY_LINES:]

    answer = reply(model, "\n".join(history) + "\n")
    print(answer)

    history.extend(answer.split("\n"))
    history = history[-MAX_HISTORY_LINES:]