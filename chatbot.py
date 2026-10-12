import torch
from bot import BigramLanguageModel
from bot import decode
from bot import encode
from bot import device
from bot import checkpoint_file

checkpoint = torch.load(checkpoint_file, map_location=device)
max_new_tokens = 200

model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

conversation = ""

while True:
    user_input = input("\n<user_1> ")
    if user_input.lower() == "exit":
        break
    conversation += f"<user_1> {user_input}\n"
    context = torch.tensor([encode(conversation)],dtype=torch.long,device=device)
    output = model.generate(context, max_new_tokens)
    generated = decode(output[0].tolist())
    response = generated[len(conversation):]
    if "<user_1>" in response:
        response = response.split("<user_1>")[0]
    print("", response)

    conversation += response + "\n"