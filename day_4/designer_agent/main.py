from .prompts import SYSTEM_PROMPT_DIALOG, SYSTEM_PROMPT_ONESHOT
from .spinner import Spinner
from .api import call_turn, call_brief

def main():
    print('🎨 Дизайнер интерьеров:\n«Здравствуйте! Я помогу создать интерьер вашей мечты. Расскажите о квартире и ожиданиях — начнём с основных пожеланий.»\n')
    history=[{'role':'system','content':SYSTEM_PROMPT_DIALOG}]
    while True:
        try:
            first=input('> ').strip()
        except EOFError:
            return
        if first.lower() in ('выход','exit','quit'): print('🪞 Дизайнер интерьеров кивает: «До встречи!»'); return
        if first:
            history.append({'role':'user','content':first}); break
    assistant_ask_turns=0; MAX_ASK_TURNS=10
    try:
        while True:
            spin=Spinner(); spin.start()
            try:
                turn=call_turn(history)
            except Exception as e:
                spin.stop_and_clear(); print(str(e))
                user=input('> ').strip()
                if user.lower() in ('выход','exit','quit'): print('🪞 Дизайнер интерьеров кивает: «До встречи!»'); break
                if user: history.append({'role':'user','content':user}); continue
            spin.stop_and_clear()
            print(turn.message or '(сообщение)')
            history.append({'role':'assistant','content':turn.message})
            if turn.mode=='final':
                if turn.brief:
                    turn.brief.pretty_print()
                    user_description=(f"Имя клиента: {turn.brief.client_name}\n"f"Площадь: {turn.brief.apartment_area}\n"f"Комнаты и назначение: {turn.brief.rooms}\n"f"Стиль: {turn.brief.style}\n"f"Палитра: {turn.brief.color_palette}\n"f"Материалы: {turn.brief.materials}\n"f"Мебель: {turn.brief.furniture}\n"f"Освещение: {turn.brief.lighting}\n"f"Бюджет: {turn.brief.budget}\n"f"Особые пожелания: {turn.brief.special_requests}\n\nСформируй полноценный дизайн-бриф по этим требованиям.")
                    for t in [0.0,0.7,1.2]:
                        sub=Spinner(f'Готовлю вариант при температуре {t}'); sub.start()
                        try:
                            variant=call_brief(SYSTEM_PROMPT_ONESHOT,user_description,temperature=t)
                        except Exception:
                            sub.stop_and_clear(); print(f"\n🌡️ Температура {t}: Попробуй ещё раз!"); continue
                        sub.stop_and_clear()
                        print('\n'+'='*80); print(f'🌡️  Вариант при температуре: {t}'); print('='*80)
                        print(f'Стиль: {variant.style}')
                        print(f'Палитра: {variant.color_palette}')
                        print(f'Материалы: {variant.materials}')
                        print(f'Освещение: {variant.lighting}')
                        print(f'Бюджет: {variant.budget}')
                        print(f'Особые пожелания: {variant.special_requests}')
                        print('\n— — — Развёрнутое видение — — —'); print(variant.final_design_brief)
                    print('\n'+'='*80); print('🧭 Как интерпретировать температуру:')
                    print('- 0.0 — максимальная точность и повторяемость формулировок; хорошо для строгого ТЗ.')
                    print('- 0.7 — баланс точности и разнообразия; хорошо для вариантов и мягкой креативности.')
                    print('- 1.2 — высокая креативность и смелые идеи; возможны вольные интерпретации.')
                    print('='*80+'\n')
                else:
                    print('Попробуй ещё раз!')
                break
            if turn.mode=='ask':
                assistant_ask_turns+=1
                user=input('> ').strip()
                if not user: history.append({'role':'user','content':'(пользователь не ответил)'})
                elif user.lower() in ('выход','exit','quit'): print('🪞 Дизайнер интерьеров кивает: «До встречи!»'); break
                else: history.append({'role':'user','content':user})
                if assistant_ask_turns>=MAX_ASK_TURNS:
                    history.append({'role':'system','content':"Больше уточнений не требуются. Сформируй следующий ход обязательно с mode='final' и полностью заполни brief."})
            else:
                print('Попробуй ещё раз!'); continue
    except KeyboardInterrupt:
        print('\n🪞 Дизайнер интерьеров прощается. Пока!')
