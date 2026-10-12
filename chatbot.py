import torch
from bot import BigramLanguageModel
from bot import decode
from bot import encode
from bot import device
from bot import checkpoint_file

checkpoint = torch.load(checkpoint_file, map_location=device)
max_new_tokens = 200
temperature = 0.65
top_k = 15

model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

conversation = ""

while True:
    user_input = input("\n<user_1> ").strip()
    if user_input.lower() == "exit":
        break

    conversation += f"<user_1> {user_input}\n<friend> "

    context = torch.tensor(
        [encode(conversation)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    new_tokens = output[0, context.shape[1]:].tolist()
    generated = decode(new_tokens)

    response = generated.split("\n", 1)[0].strip()

    if not response:
        response = "Hmm, give me a second to think."

    print("<friend>", response)
    conversation += response + "\n"