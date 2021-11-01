# import requests
import asyncio
from aiogram import Bot, Dispatcher, types
import pickle
# import transformers
# import torch

token = ''
url_bot = f"https://api.telegram.org/bot{token}/"

with open('/home/ishubenko/Загрузки/nlp/tokenizer_medium.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('/home/ishubenko/Загрузки/nlp/model_medium.pickle', 'rb') as handle:
    model = pickle.load(handle)

def respond_to_dialog(texts):
    prefix = '\nx:'
    for i, t in enumerate(texts):
        prefix += t
        prefix += '\nx:' if i % 2 == 1 else '\ny:'
    tokens = tokenizer(prefix, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    end_token_id = tokenizer.encode('\n')[0]
    size = tokens['input_ids'].shape[1]
    output = model.generate(
        **tokens,
        eos_token_id=end_token_id,
        do_sample=True,
        max_length=size+128,
        repetition_penalty=3.2,
        temperature=1,
        num_beams=3,
        length_penalty=0.01,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0])
    result = decoded[len(prefix):]
    return result.strip()

def dialog(question):
    # question = 'Чем занимаешься?'
    history = [question]
    result = respond_to_dialog(history[-10:])
    return result

# def dialog(text):
#     return text[:-1]

async def start_handler(event: types.Message):
    await event.answer(
        f"Привет, {event.from_user.get_mention(as_html=True)}",
        parse_mode=types.ParseMode.HTML,
    )

async def simple_message(event: types.Message):
    await event.answer(
        dialog(event.html_text)
    )

async def main():
    bot = Bot(token=token)
    try:
        disp = Dispatcher(bot=bot)
        disp.register_message_handler(start_handler, commands={"start", "restart"})
        disp.register_message_handler(simple_message)
        await disp.start_polling()
    finally:
        await bot.close()

asyncio.run(main())

# response = requests.post(url_bot + 'getUpdates') #, data={'chat_id': '-483165409', 'text': 'приветосики абрикосики'})

print(1)