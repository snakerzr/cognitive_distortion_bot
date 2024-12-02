from openai import OpenAI

import logging

logger = logging.getLogger('bot.services.openai')


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def get_response(
        self,
        user_message: str,
        system_prompt: str,
        response_schema=None,
        model="gpt-4o-mini",
        temperature=None,
    ):
        try:
            request_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }

            if temperature:
                request_payload["temperature"] = temperature

            if response_schema:
                completion = self.client.beta.chat.completions.parse(
                    **request_payload,
                    response_format=response_schema,
                )
                return completion.choices[0].message.parsed  # Возвращает анализ со схемой
            else:
                completion = self.client.chat.completions.create(**request_payload)
                return completion.choices[0].message.content  # Возвращает текст без схемы
        except Exception as e:
            print(f"Ошибка при вызове OpenAI API: {e}")
            return None





# from openai import OpenAI
# from src.config import config
# from src.services.openai.schema import (
#     DistortionSchema,
#     CatastrophizingSchema,
#     OvergeneralizationSchema,
#     MindReadingSchema,
#     BlackAndWhiteThinkingSchema,
#     EmotionalReasoningSchema,
#     PersonalizationSchema,
#     ShouldStatementsSchema,
# )

# # Инициализация OpenAI API
# client = OpenAI(api_key=config.OPENAI_API_KEY.get_secret_value())


# def analyze_thought(
#     user_message: str, system_prompt: str, distortion_schema: DistortionSchema
# ) -> DistortionSchema:
#     try:
#         completion = client.beta.chat.completions.parse(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message},
#             ],
#             response_format=distortion_schema,
#         )

#         event = completion.choices[0].message
#         if event.refusal:
#             return event.refusal  # Возвращаем причину отказа
#         else:
#             return event.parsed  # Возвращаем успешный результат анализа
#     except Exception as e:
#         print(f"Ошибка при анализе мысли: {e}")
#         return None


# def analyze_all_distortions(
#     user_message: str,
#     system_prompt: str = "Проверьте это утверждение на когнитивные искажения.",
# ) -> list[DistortionSchema]:
#     # Словарь всех схем с названием как ключом и классом схемы как значением
#     schemas_dict = {
#         "Катастрофизация": CatastrophizingSchema,
#         "Обобщение": OvergeneralizationSchema,
#         "Чтение мыслей": MindReadingSchema,
#         "Черно-белое мышление": BlackAndWhiteThinkingSchema,
#         "Эмоциональное обоснование": EmotionalReasoningSchema,
#         "Персонализация": PersonalizationSchema,
#         "Долженствования": ShouldStatementsSchema,
#     }

#     results = []

#     for name, schema_class in schemas_dict.items():
#         result_dict = {}
#         result_dict[name] = analyze_thought(user_message, system_prompt, schema_class)
#         results.append(result_dict)

#     return results


# def is_valid_thought(*args) -> bool:
#     return True


# if __name__ == "__main__":
#     user_message = "Я думаю, что всё всегда идет не так, как должно."
#     system_prompt = "Проверьте это утверждение на когнитивные искажения."

#     # Получение полного результата анализа
#     results = analyze_all_distortions(user_message, system_prompt)

#     for result in results:
#         for name, schema_result in result.items():
#             print(f"{name}:")
#             try:
#                 print(f"Присутствие искажения: {schema_result.present}")
#                 print(f"Обоснование: {schema_result.reasoning}\n")
#             except:
#                 pass
